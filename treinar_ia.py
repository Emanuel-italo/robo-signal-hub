import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score

# === CONFIGURAÇÕES ===
# Lista expandida para 40 ativos
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
    'AVAX/USDT', 'DOT/USDT', 'TRX/USDT', 'LINK/USDT', 'MATIC/USDT', 'ATOM/USDT',
    'NEAR/USDT', 'ALGO/USDT', 'ICP/USDT', 'FTM/USDT', 'APT/USDT', 'SUI/USDT',
    'ARB/USDT', 'OP/USDT', 'IMX/USDT',
    'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT',
    'UNI/USDT', 'AAVE/USDT', 'LTC/USDT', 'BCH/USDT', 'ETC/USDT', 'FIL/USDT',
    'RNDR/USDT', 'FET/USDT', 'GRT/USDT', 'SAND/USDT', 'MANA/USDT', 'AXS/USDT',
    'GALA/USDT'
]

TIMEFRAME = '1h'
LIMIT = 1500 

# === CONFIGURAÇÃO DE PASTA (GOOGLE DRIVE) ===
caminhos_possiveis = [
    r'G:\Meu Drive', r'G:\My Drive', 
    r'D:\Meu Drive', r'E:\Meu Drive',
    os.path.expanduser('~/Google Drive'),
    os.getcwd()
]
# Tenta achar o Drive, senão usa pasta local
BASE_DRIVE = next((path for path in caminhos_possiveis if os.path.exists(path)), os.getcwd())
PASTA_PROJETO = os.path.join(BASE_DRIVE, 'RoboTrader_Arquivos_final')
META_DIR = os.path.join(PASTA_PROJETO, 'model_metadata')
INTEL_DIR = os.path.join(PASTA_PROJETO, 'inteligencia_ia')

os.makedirs(META_DIR, exist_ok=True)
os.makedirs(INTEL_DIR, exist_ok=True)

print(f"📂 ARMAZENAMENTO DEFINIDO: {PASTA_PROJETO}")

exchange = ccxt.binance()

def calcular_indicadores(df):
    """
    Cria os indicadores técnicos com PROTEÇÃO contra erros de nomes de colunas.
    """
    df = df.copy()
    
    # 1. Tendência e Momento
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # MACD Seguro
    macd = ta.macd(df['close'])
    macd_col = [c for c in macd.columns if c.startswith('MACD_') and not c.startswith('MACDs') and not c.startswith('MACDh')]
    if macd_col:
        df['macd'] = macd[macd_col[0]]
    else:
        df['macd'] = 0 
    
    df['ema_50'] = ta.ema(df['close'], length=50)
    
    # 2. Volatilidade (ATR)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Bollinger Bands Seguro
    bb = ta.bbands(df['close'], length=20, std=2)
    bbu_col = [c for c in bb.columns if c.startswith('BBU')]
    bbl_col = [c for c in bb.columns if c.startswith('BBL')]
    
    if bbu_col and bbl_col:
        df['bb_upper'] = bb[bbu_col[0]]
        df['bb_lower'] = bb[bbl_col[0]]
    else:
        df['bb_upper'] = df['close'] 
        df['bb_lower'] = df['close']

    # 3. Features Relativas
    df['dist_ema'] = (df['close'] - df['ema_50']) / df['ema_50']
    
    df['dist_bb_up'] = 0.0
    mask = df['close'] != 0
    df.loc[mask, 'dist_bb_up'] = (df.loc[mask, 'bb_upper'] - df.loc[mask, 'close']) / df.loc[mask, 'close']
    
    df['rsi_norm'] = df['rsi'] / 100
    df['vol_change'] = df['volume'].pct_change()

    return df

def treinar_simbolo(symbol):
    print(f"--- Treinando IA para {symbol} ---")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        print(f"Erro ao baixar {symbol}: {e}")
        return

    if len(df) < 200:
        print(f"Dados insuficientes para {symbol}")
        return

    try:
        df = calcular_indicadores(df)
    except Exception as e:
        print(f"Erro indicadores {symbol}: {e}")
        return

    # ALVO: Subir > 1% nas próximas 3 horas
    future_close = df['close'].shift(-3) 
    df['target'] = (future_close > df['close'] * 1.01).astype(int)
    df = df.dropna()

    features = ['rsi_norm', 'dist_ema', 'dist_bb_up', 'atr', 'vol_change', 'macd']
    X = df[features]
    y = df['target']

    if y.sum() < 10:
        print(f"{symbol}: Poucas oportunidades. Pulando.")
        return

    split = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    
    print(f"Resultado {symbol} -> Acurácia: {acc:.2f} | Precisão: {precision:.2f}")

    if acc > 0.51:
        safe_sym = symbol.replace('/', '_')
        joblib.dump(model, os.path.join(INTEL_DIR, f"{safe_sym}.joblib"))
        meta = {
            "features": features,
            "accuracy": acc,
            "trained_at": str(pd.Timestamp.now())
        }
        with open(os.path.join(META_DIR, f"{safe_sym}_meta.json"), 'w') as f:
            json.dump(meta, f)
        print(f">> {symbol} SALVO NO DRIVE <<")
    else:
        print(f">> {symbol} DESCARTADO <<")

if __name__ == "__main__":
    for sym in SYMBOLS:
        treinar_simbolo(sym)