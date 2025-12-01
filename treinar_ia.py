# train_models_improved.py
# -*- coding: utf-8 -*-
"""
Treinamento OTIMIZADO para salvar modelos funcionais.
Mudanças Críticas:
 - Alvo (Target) reduzido para 0.6% para encontrar mais oportunidades.
 - Critério de salvamento baseado em LUCRO (ROI) no backtest, não apenas acurácia teórica.
 - Diagnóstico de balanceamento de classes.
"""
from __future__ import annotations

import os
import json
import time
import math
import joblib
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    precision_recall_curve, roc_auc_score
)
from typing import List

# ------------------------------
# CONFIG
# ------------------------------
TIMEFRAME = '1h'
CHUNK_LIMIT = 1500
HIST_YEARS = 3

# CONFIGURAÇÃO DE DIRETÓRIOS
caminhos_possiveis = [
    r'G:\Meu Drive', r'G:\My Drive', r'D:\Meu Drive', r'E:\Meu Drive',
    os.path.expanduser('~/Google Drive'), os.getcwd()
]
BASE_DRIVE = next((path for path in caminhos_possiveis if os.path.exists(path)), os.getcwd())
PASTA_PROJETO = os.path.join(BASE_DRIVE, 'RoboTrader_Arquivos_final')
META_DIR = os.path.join(PASTA_PROJETO, 'model_metadata')
INTEL_DIR = os.path.join(PASTA_PROJETO, 'inteligencia_ia')

os.makedirs(META_DIR, exist_ok=True)
os.makedirs(INTEL_DIR, exist_ok=True)

print(f"📂 ARMAZENAMENTO: {PASTA_PROJETO}")

# LISTA DE MOEDAS (Sua lista completa)
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
    'AVAX/USDT', 'DOT/USDT', 'TRX/USDT', 'LINK/USDT', 'MATIC/USDT', 'ATOM/USDT',
    'NEAR/USDT', 'ALGO/USDT', 'ICP/USDT', 'FTM/USDT', 'APT/USDT', 'SUI/USDT',
    'ARB/USDT', 'OP/USDT', 'IMX/USDT',
    'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT',
    'UNI/USDT', 'AAVE/USDT', 'LTC/USDT', 'BCH/USDT', 'ETC/USDT', 'FIL/USDT',
    'RNDR/USDT', 'FET/USDT', 'GRT/USDT', 'SAND/USDT', 'MANA/USDT', 'AXS/USDT',
    'GALA/USDT'
] + [
    'TWT/USDT','ANKR/USDT','RVN/USDT','ZIL/USDT','HNT/USDT','ENJ/USDT','SUSHI/USDT','YFI/USDT',
    'CELR/USDT','ONE/USDT','IOTA/USDT','NANO/USDT','QTUM/USDT','HOT/USDT','NMR/USDT','CHZ/USDT',
    'IOST/USDT','SRM/USDT','ORN/USDT','OCEAN/USDT','BAT/USDT','KAVA/USDT','ZEN/USDT','KNC/USDT',
    'ZRX/USDT','XEM/USDT','STX/USDT','SXP/USDT','HIVE/USDT','RLC/USDT','BAND/USDT','MTL/USDT',
    'LINA/USDT','ALPHA/USDT','LRC/USDT','SFP/USDT','FIDA/USDT','TLM/USDT','ARPA/USDT','OGN/USDT'
]

# ESTRATÉGIA AJUSTADA
TARGET_HORIZON = 3           # horas à frente
MIN_GAIN_PCT = 0.006         # 0.6% (Reduzido de 1% para facilitar aprendizado)
SLIPPAGE_PCT = 0.0015        # 0.15%
FEE_PCT = 0.001              # 0.1%

# Random seed
SEED = 42

# ------------------------------
# UTIL / EXTRACÇÃO
# ------------------------------
exchange = ccxt.binance({'enableRateLimit': True})

def timeframe_to_ms(tf: str) -> int:
    unit = tf[-1]
    val = int(tf[:-1])
    if unit == 'm': return val * 60 * 1000
    if unit == 'h': return val * 60 * 60 * 1000
    if unit == 'd': return val * 24 * 60 * 60 * 1000
    if unit == 'w': return val * 7 * 24 * 60 * 60 * 1000
    return val * 60 * 1000

def fetch_ohlcv_full(symbol: str, timeframe: str = TIMEFRAME, years: int = HIST_YEARS, chunk_limit: int = CHUNK_LIMIT):
    try:
        ms_per_candle = timeframe_to_ms(timeframe)
        now_ms = int(time.time() * 1000)
        since_ms = now_ms - int(years * 365 * 24 * 60 * 60 * 1000)
        all_rows = []
        last_ts = None
        attempt = 0
        while True:
            attempt += 1
            try:
                rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=chunk_limit)
            except Exception:
                if attempt <= 2:
                    time.sleep(1)
                    continue
                raise

            if not rows:
                break

            all_rows.extend(rows)
            last_ts = rows[-1][0]
            next_since = int(last_ts + ms_per_candle + 1)
            if next_since <= since_ms:
                break
            since_ms = next_since
            if last_ts >= now_ms - ms_per_candle:
                break
            if len(all_rows) > (chunk_limit * 150): # Limite de segurança
                break
            time.sleep(0.1) # Rápido, mas respeitoso
        
        if not all_rows:
            return None
        df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[fetch_full] Erro {symbol}: {e}")
        return None

def safe_fetch_ohlcv(symbol: str, timeframe: str = TIMEFRAME, limit: int = CHUNK_LIMIT):
    return fetch_ohlcv_full(symbol, timeframe=timeframe, years=HIST_YEARS, chunk_limit=CHUNK_LIMIT)

# ------------------------------
# FEATURES
# ------------------------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ('c', 'close'): colmap[c] = 'close'
        elif lc in ('o', 'open'): colmap[c] = 'open'
        elif lc in ('h', 'high'): colmap[c] = 'high'
        elif lc in ('l', 'low'): colmap[c] = 'low'
        elif lc in ('v', 'volume'): colmap[c] = 'volume'
    if colmap:
        df = df.rename(columns=colmap)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].replace([np.inf, -np.inf], np.nan)

    # Indicadores
    try:
        df['rsi'] = ta.rsi(df['close'], length=14)
    except: df['rsi'] = np.nan

    try:
        macd = ta.macd(df['close'])
        if isinstance(macd, pd.DataFrame) and not macd.empty:
            df['macd'] = macd.iloc[:, 0]
        else: df['macd'] = np.nan
    except: df['macd'] = np.nan

    try:
        df['ema_50'] = ta.ema(df['close'], length=50)
    except: df['ema_50'] = df['close']

    try:
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    except: df['atr'] = np.nan
    
    # Fallback ATR
    df['atr'] = df['atr'].replace(0, np.nan)
    df['atr'] = df['atr'].fillna(df['close'] * 0.005)

    try:
        bb = ta.bbands(df['close'], length=20, std=2)
        if isinstance(bb, pd.DataFrame) and not bb.empty:
            df['bb_upper'] = bb.iloc[:, 0]
        else: df['bb_upper'] = df['close']
    except: df['bb_upper'] = df['close']

    # Features derivadas
    df['dist_ema'] = (df['close'] - df['ema_50']) / df['ema_50'].replace(0, np.nan)
    df['dist_bb_up'] = ((df['bb_upper'] - df['close']) / df['close']).replace([np.inf, -np.inf], 0).fillna(0)
    df['rsi_norm'] = df['rsi'] / 100.0
    
    try:
        df['vol_change'] = df['volume'].pct_change().fillna(0)
    except: df['vol_change'] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

# ------------------------------
# TARGET
# ------------------------------
def make_target(df: pd.DataFrame, horizon: int = TARGET_HORIZON, min_gain_pct: float = MIN_GAIN_PCT) -> pd.Series:
    future_close = df['close'].shift(-horizon)
    # Considera fees+slippage
    threshold_price = df['close'] * (1.0 + min_gain_pct + SLIPPAGE_PCT + FEE_PCT)
    target = (future_close >= threshold_price).astype(int)
    return target

# ------------------------------
# BACKTEST
# ------------------------------
def backtest_signals(df: pd.DataFrame, preds_proba: np.ndarray, threshold: float, allocation_usd: float = 10.0):
    horizon = TARGET_HORIZON
    entries = preds_proba >= threshold
    results = []
    
    # Iterar apenas onde é possível fechar o trade
    for t in range(len(df) - horizon):
        if entries[t]:
            entry_price = df['close'].iloc[t] * (1 + SLIPPAGE_PCT)
            exit_price = df['close'].iloc[t + horizon] * (1 - SLIPPAGE_PCT) # Vende X horas depois
            
            if pd.isna(exit_price) or entry_price <= 0: continue

            fee_cost = (entry_price + exit_price) * FEE_PCT
            qty = allocation_usd / entry_price
            pnl = (exit_price - entry_price) * qty - fee_cost
            pnl_pct = (pnl / allocation_usd) * 100
            
            win = pnl > 0
            results.append({'pnl': pnl, 'pnl_pct': pnl_pct, 'win': win})

    if not results:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'roi_pct': 0.0}

    total_pnl = sum([r['pnl'] for r in results])
    wins = sum([1 for r in results if r['win']])
    losses = len(results) - wins
    # ROI simples sobre o capital alocado (não composto)
    roi_pct = (total_pnl / (len(results) * allocation_usd)) * 100 
    
    return {'trades': len(results), 'wins': wins, 'losses': losses, 'total_pnl': total_pnl, 'roi_pct': roi_pct}

# ------------------------------
# OVERSAMPLE
# ------------------------------
def simple_oversample(X: pd.DataFrame, y: pd.Series, target_ratio: float = 0.4):
    if len(y) == 0: return X, y
    counts = y.value_counts()
    if len(counts) < 2: return X, y
    
    maj_count = counts.max()
    min_count = counts.min()
    min_label = counts.idxmin()

    desired_min = int((target_ratio / (1.0 - target_ratio)) * maj_count)
    if min_count >= desired_min: return X, y

    min_idx = y[y == min_label].index.to_numpy()
    needed = desired_min - min_count
    
    # Cap no tamanho para não explodir memória
    if needed > len(X) * 2: needed = len(X)

    choices = np.random.choice(min_idx, size=needed, replace=True)
    X_extra = X.loc[choices].copy()
    y_extra = y.loc[choices].copy()
    
    X_res = pd.concat([X, X_extra], ignore_index=True)
    y_res = pd.concat([y, y_extra], ignore_index=True)
    return X_res.reset_index(drop=True), y_res.reset_index(drop=True)

# ------------------------------
# TREINO POR SÍMBOLO (Lógica Nova)
# ------------------------------
def treinar_simbolo(symbol: str):
    print(f"\n--- Treinando: {symbol} ---")
    df = safe_fetch_ohlcv(symbol, TIMEFRAME, CHUNK_LIMIT)
    if df is None or len(df) < 500:
        print(f"[{symbol}] Dados insuficientes.")
        return False

    try:
        df = calcular_indicadores(df)
    except Exception as e:
        print(f"[{symbol}] Erro feature eng: {e}")
        return False

    df['target'] = make_target(df)
    df = df.dropna().reset_index(drop=True)

    # === DIAGNÓSTICO DE CLASSES ===
    pos = df['target'].sum()
    total = len(df)
    ratio = pos / total if total > 0 else 0
    print(f"[{symbol}] Amostras: {total} | Positivos: {pos} ({ratio:.2%})")

    if pos < 50:
        print(f"[{symbol}] ⚠️ Pular: Pouquíssimos exemplos positivos para aprender.")
        return False

    features = ['rsi_norm', 'dist_ema', 'dist_bb_up', 'atr', 'vol_change', 'macd']
    X = df[features].fillna(0.0)
    y = df['target'].astype(int)

    # Split
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Oversample
    X_train_res, y_train_res = simple_oversample(X_train, y_train, target_ratio=0.35)

    # Modelo (Simples e rápido para teste)
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=4, 
                                random_state=SEED, class_weight='balanced', n_jobs=-1)
    
    try:
        rf.fit(X_train_res, y_train_res)
    except:
        return False

    # Predict Probabilities
    try:
        y_proba = rf.predict_proba(X_test)[:, 1]
    except:
        y_proba = np.zeros(len(X_test))

    # --- Threshold Dinâmico ---
    # Busca um threshold que tenha precisão > 52% e maximize recall
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Filtra thresholds com precisão aceitável (ex: > 0.51)
    # Precisamos alinhar arrays (thresholds é menor que precision em 1)
    valid_indices = [i for i, p in enumerate(precisions[:-1]) if p >= 0.51]
    
    if valid_indices:
        # Pega o índice que dá o MAIOR recall dentre os que tem precisão boa
        # Como thresholds sobem, recall desce. O primeiro índice válido geralmente tem o maior recall.
        best_idx = valid_indices[0] 
        chosen_threshold = thresholds[best_idx]
    else:
        # Se nenhum atingiu 51% de precisão, pega o de máxima precisão ou padrão
        if len(thresholds) > 0:
            chosen_threshold = thresholds[np.argmax(precisions[:-1])]
        else:
            chosen_threshold = 0.5

    # Segurança: não deixar threshold absurdo
    chosen_threshold = max(0.4, min(chosen_threshold, 0.85))

    # Gera classes finais para métricas
    y_pred_final = (y_proba >= chosen_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_final)
    prec = precision_score(y_test, y_pred_final, zero_division=0)
    rec = recall_score(y_test, y_pred_final, zero_division=0)

    # Backtest Rápido
    bt = backtest_signals(df.iloc[split_idx:].reset_index(drop=True), y_proba, chosen_threshold)

    print(f"[{symbol}] Threshold: {chosen_threshold:.3f} | Prec: {prec:.2f} | Trades: {bt['trades']} | ROI: {bt['roi_pct']:.2f}%")

    # === CRITÉRIO DE SALVAMENTO RELAXADO ===
    # Opção A: Precisão decente e Lucro positivo com volume mínimo
    cond_a = (prec >= 0.50 and bt['roi_pct'] > 0 and bt['trades'] >= 3)
    # Opção B: Lucro muito bom (>4%) independente da precisão técnica
    cond_b = (bt['roi_pct'] >= 4.0 and bt['trades'] >= 1)

    if cond_a or cond_b:
        safe_sym = symbol.replace('/', '_')
        joblib.dump(rf, os.path.join(INTEL_DIR, f"{safe_sym}.joblib"))
        
        meta = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "threshold": float(chosen_threshold),
            "roi_pct": float(bt['roi_pct']),
            "trades": bt['trades'],
            "target_gain": MIN_GAIN_PCT
        }
        with open(os.path.join(META_DIR, f"{safe_sym}_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f">> ✅ {symbol} MODELO SALVO! (Lucro detectado)")
        return True
    else:
        print(f">> ❌ {symbol} descartado (ROI ruim ou precisão baixa).")
        return False

# ------------------------------
# MAIN execution
# ------------------------------
if __name__ == "__main__":
    start = time.time()
    saved_count = 0
    tried = 0
    
    print("\n=== INICIANDO TREINAMENTO OTIMIZADO ===")
    print(f"Target de Lucro: {MIN_GAIN_PCT*100}% | Slippage: {SLIPPAGE_PCT*100}%")
    
    for sym in SYMBOLS:
        tried += 1
        try:
            ok = treinar_simbolo(sym)
            if ok: saved_count += 1
        except KeyboardInterrupt:
            print("Parando pelo usuário...")
            break
        except Exception as e:
            print(f"Erro fatal em {sym}: {e}")
    
    elapsed = (time.time() - start) / 60
    print(f"\nResumo: {saved_count} modelos salvos de {tried} testados.")
    print(f"Tempo total: {elapsed:.2f} min")