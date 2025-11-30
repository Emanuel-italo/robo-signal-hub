"""
Robô Final Integrado (Versão Turbo + API Rica + Google Drive + Logs Dinâmicos)
Adaptado para salvar arquivos preferencialmente no Google Drive.
CORREÇÃO: Suporte a UTF-8 para emojis no Windows.
"""

import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import feedparser
import time
import os
import json
import logging
import threading
from datetime import datetime, timedelta, date
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import warnings

# --- Bibliotecas para API ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Rich for colored logs
from rich.logging import RichHandler
import rich.traceback
rich.traceback.install()

warnings.filterwarnings("ignore")

# -------------------------- SISTEMA DE LOGS PARA WEB -----------------
log_buffer = []
last_known_prices = {}  # Guarda o preço em tempo real para o Front
last_trade_event = None # Guarda a última ação para notificar o Front

class WebLogger(logging.Handler):
    def emit(self, record):
        try:
            log_entry = self.format(record)
            timestamp = datetime.now().strftime('%H:%M:%S')
            level = record.levelname
            log_buffer.insert(0, {
                "time": timestamp, 
                "level": level, 
                "message": record.getMessage()
            })
            if len(log_buffer) > 200:
                log_buffer.pop()
        except Exception:
            self.handleError(record)

# -------------------------- CONFIGURAÇÃO DE DRIVE --------------------
caminhos_possiveis = [
    r'G:\Meu Drive',                 
    r'G:\My Drive',                  
    r'D:\Meu Drive',                 
    os.path.expanduser('~/Google Drive'), 
    os.getcwd()                      
]

BASE_DRIVE = next((path for path in caminhos_possiveis if os.path.exists(path)), os.getcwd())
PASTA_PROJETO = os.path.join(BASE_DRIVE, 'RoboTrader_Arquivos_final')

try:
    os.makedirs(PASTA_PROJETO, exist_ok=True)
    print(f"\n[SISTEMA] ARQUIVOS SERÃO SALVOS EM: {PASTA_PROJETO}\n")
except Exception as e:
    print(f"[ERRO] Não foi possível criar pasta no Drive. Usando local. Erro: {e}")
    BASE_DRIVE = os.getcwd()
    PASTA_PROJETO = os.path.join(BASE_DRIVE, 'RoboTrader_Arquivos_final')
    os.makedirs(PASTA_PROJETO, exist_ok=True)

CONFIG = {
    # --- MERCADO (20+ Moedas) ---
    'symbols': [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
        'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'TRX/USDT', 'DOT/USDT',
        'LINK/USDT', 'MATIC/USDT', 'LTC/USDT', 'SHIB/USDT', 'UNI/USDT',
        'ATOM/USDT', 'XMR/USDT', 'ETC/USDT', 'FIL/USDT', 'APT/USDT','LSK/USDT'
    ],
    'timeframe': '1h',
    # capital / risk
    'saldo_inicial_ficticio': 10000.0,
    'risco_por_trade': 0.01,            
    'fixed_fraction_invest': 0.02,      
    'investiment_mode': 'fixed_fraction', 
    # stops dinamicos
    'atr_stop_mult': 1.0, 
    'atr_take_mult': 2.0, 
    'stop_loss_pct': 0.02, 
    'take_profit_pct': 0.04,
    'min_invest_usd': 10.0,
    # custos
    'fee_pct': 0.001,   # 0.1%
    'slippage_pct': 0.001,
    'min_profit_margin': 0.003,
    # modelo / labeling
    'lookahead_candles': 24, 
    'target_quantile': 0.75, 
    'target_quantile_grid': [0.6, 0.7, 0.75, 0.8],
    # operacional
    'retrain_interval_minutes': 24*60, 
    'enable_live_trading': False,        
    'min_oos_accuracy': 0.55, 
    'confianca_minima': 0.50, 
    'vwap_window': 24, 
    'vwap_trend_tolerance': 0.995, 
    # gestão de risco extra
    'max_daily_trades': 10,  # Aumentado para 10
    'max_exposure_pct': 0.50, # Aumentado para 50%
    # infra
    'training_csv_dir': os.path.join(PASTA_PROJETO, 'dados_historicos'),
    'trades_log_csv': os.path.join(PASTA_PROJETO, 'relatorio_trades.csv'),
    'models_dir': os.path.join(PASTA_PROJETO, 'inteligencia_ia'),
    'feature_imp_dir': os.path.join(PASTA_PROJETO, 'feature_importances'),
    'model_metadata_dir': os.path.join(PASTA_PROJETO, 'model_metadata'),
    # treinamento
    'n_estimators_candidates': [50, 100, 150],
    'n_jobs_search': -1,
    'warm_start': False,
    # anti-ban / throttle (OTIMIZADO)
    'sleep_between_symbol_calls': 0.5, # Mais rápido
    'sleep_before_ohlcv_call': 0.2,
    # misc
    'log_file': os.path.join(PASTA_PROJETO, 'robo_log.txt'),
    'max_models_to_train_simultaneously': 3
}

RSS_FEEDS = [
    'https://cryptopanic.com/news/rss/',
    'https://cointelegraph.com/rss',
    'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'https://finance.yahoo.com/news/rssindex',
    'https://decrypt.co/feed',
    'https://cryptoslate.com/feed/',
    'https://dailyhodl.com/feed/',
    'https://bitcoinmagazine.com/feed',
    'http://feeds.reuters.com/reuters/businessNews'
]

# -------------------------- PREPARA PASTAS E LOG ---------------------
os.makedirs(CONFIG['training_csv_dir'], exist_ok=True)
os.makedirs(CONFIG['models_dir'], exist_ok=True)
os.makedirs(CONFIG['feature_imp_dir'], exist_ok=True)
os.makedirs(CONFIG['model_metadata_dir'], exist_ok=True)

logger = logging.getLogger('RoboTraderFinal')
logger.setLevel(logging.DEBUG)

# Handlers - CORREÇÃO APLICADA AQUI (encoding='utf-8')
fh = logging.FileHandler(CONFIG['log_file'], encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

rh = RichHandler(rich_tracebacks=True)
rh.setLevel(logging.INFO)

wh = WebLogger()
wh.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(rh)
    logger.addHandler(wh)

# -------------------------- UTILITÁRIOS ------------------------------

def now_str():
    return datetime.utcnow().strftime('%Y%m%d_%H%M%S')

def salvar_trade_csv(trade_dict, filename=CONFIG['trades_log_csv']):
    tentativas = 0
    while tentativas < 8:
        try:
            df = pd.DataFrame([trade_dict])
            header = not os.path.exists(filename)
            df.to_csv(filename, mode='a', index=False, header=header)
            return True
        except PermissionError:
            time.sleep(1)
            tentativas += 1
        except Exception as e:
            logger.error(f"salvar_trade_csv unexpected error: {e}")
            break
    return False

def model_path(symbol, suffix=None):
    name = symbol.replace('/','_')
    if suffix: return os.path.join(CONFIG['models_dir'], f"{name}_{suffix}.joblib")
    return os.path.join(CONFIG['models_dir'], f"{name}.joblib")

def metadata_path(symbol):
    name = symbol.replace('/','_')
    return os.path.join(CONFIG['model_metadata_dir'], f"{name}_meta.json")

def save_feature_importances(symbol, features, importances):
    try:
        df = pd.DataFrame({'feature': features, 'importance': importances})
        df.sort_values('importance', ascending=False, inplace=True)
        path = os.path.join(CONFIG['feature_imp_dir'], f'feature_importances_{symbol.replace("/","_")}.csv')
        df.to_csv(path, index=False)
    except Exception as e:
        logger.warning(f"Erro save_feature_importances: {e}")

def compute_backtest_metrics(equity_series):
    if len(equity_series) < 2: return {'cum_return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}
    returns = equity_series.pct_change().fillna(0)
    cum_return = float(equity_series.iloc[-1] / equity_series.iloc[0] - 1)
    sharpe = float((returns.mean()/returns.std())*np.sqrt(252)) if returns.std() != 0 else 0.0
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max)/roll_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    return {'cum_return': cum_return, 'sharpe': sharpe, 'max_drawdown': max_dd}

# -------------------------- CARTEIRA SIMULADA ------------------------
class CarteiraVirtual:
    def __init__(self, saldo_inicial):
        self.saldo = float(saldo_inicial)
        self.posicoes = {} 
        self.trades_fechados = []
        self.trades_today = []

    def _prune_trades_today(self):
        today = date.today()
        self.trades_today = [t for t in self.trades_today if t.date() == today]

    def calcular_lucro_flutuante(self, symbol, preco_atual):
        if symbol not in self.posicoes: return 0.0
        d = self.posicoes[symbol]
        return (d['quantidade'] * preco_atual) - d['valor_investido']

    def registrar_trade_fechado(self, symbol, entry_price, exit_price, quantidade, pnl, tipo):
        global last_trade_event
        trade = {
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantidade,
            'pnl': pnl,
            'pnl_pct': (exit_price - entry_price)/entry_price if entry_price!=0 else 0.0,
            'tipo': tipo,
            'saldo_atual': self.saldo
        }
        self.trades_fechados.append(trade)
        self.trades_today.append(datetime.utcnow())
        salvar_trade_csv(trade)
        
        # DISPARO DE NOTIFICAÇÃO DE VENDA
        last_trade_event = {'symbol': symbol, 'type': 'SELL', 'pnl': pnl}

    def verificar_saidas(self, symbol, preco_atual):
        if symbol not in self.posicoes: return False
        dados = self.posicoes[symbol]
        preco_entrada = dados['preco_entrada']
        qtd = dados['quantidade']
        stop_price = dados.get('stop_price')
        take_price = dados.get('take_price')
        tipo_saida = None
        
        if stop_price is not None and preco_atual <= stop_price:
            tipo_saida = "STOP LOSS 🛑"
        elif take_price is not None and preco_atual >= take_price:
            tipo_saida = "TAKE PROFIT 💰"
        else:
            var_pct = (preco_atual - preco_entrada)/preco_entrada
            if var_pct <= -CONFIG['stop_loss_pct']: type_saida = "STOP LOSS 🛑"
            elif var_pct >= CONFIG['take_profit_pct']: type_saida = "TAKE PROFIT 💰"
        
        if tipo_saida:
            preco_saida = preco_atual * (1 - CONFIG['slippage_pct'])
            bruto = qtd * preco_saida
            fee = bruto * CONFIG['fee_pct']
            liquido = bruto - fee
            self.saldo += liquido
            lucro = liquido - dados['valor_investido']
            logger.info(f"[FECHOU] {symbol} {tipo_saida} PnL: ${lucro:.2f}")
            self.registrar_trade_fechado(symbol, dados['preco_entrada'], preco_saida, qtd, lucro, tipo_saida)
            del self.posicoes[symbol]
            return True
        return False

    def current_exposure_pct(self):
        invested = sum([p['valor_investido'] for p in self.posicoes.values()])
        total = max(self.saldo + invested, 1e-9)
        return invested / total

    def comprar(self, symbol, market_price, score_ia, atr=None):
        global last_trade_event
        self._prune_trades_today()
        if symbol in self.posicoes: return False
        if len(self.trades_today) >= CONFIG['max_daily_trades']:
            logger.info(f"[RISK] Limite diário atingido.")
            return False
        if self.current_exposure_pct() >= CONFIG['max_exposure_pct']:
            logger.info(f"[RISK] Exposição máxima atingida.")
            return False
        
        if CONFIG['investiment_mode'] == 'fixed_fraction':
            invest = self.saldo * CONFIG['fixed_fraction_invest']
        else:
            risk_amount = self.saldo * CONFIG['risco_por_trade']
            if CONFIG['stop_loss_pct'] <= 0: return False
            invest = risk_amount / CONFIG['stop_loss_pct']
            invest = min(invest, self.saldo * 0.20)
            
        if invest < CONFIG['min_invest_usd']: return False
        
        preco_entrada = market_price * (1 + CONFIG['slippage_pct'])
        qtd = invest / preco_entrada
        fee = invest * CONFIG['fee_pct']
        if (invest + fee) > self.saldo: return False
        
        stop_price = preco_entrada - (atr * CONFIG['atr_stop_mult']) if atr else None
        take_price = preco_entrada + (atr * CONFIG['atr_take_mult']) if atr else None
            
        self.saldo -= (invest + fee)
        self.posicoes[symbol] = {
            'quantidade': qtd, 'preco_entrada': preco_entrada, 'valor_investido': invest,
            'hora': datetime.utcnow(), 'atr_at_entry': atr,
            'stop_price': stop_price, 'take_price': take_price
        }
        logger.info(f"[COMPRA] 🚀 {symbol} @ {preco_entrada:.2f} | Score: {score_ia:.2f}")
        
        # DISPARO DE NOTIFICAÇÃO DE COMPRA
        last_trade_event = {'symbol': symbol, 'type': 'BUY', 'price': preco_entrada}
        return True

# -------------------------- ROBÔ PRINCIPAL -------------------------
class RoboTrader:
    def __init__(self, paper=True):
        self.paper = paper
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.carteira = CarteiraVirtual(CONFIG['saldo_inicial_ficticio'])
        self.modelos = {} 
        self.analyzer = SentimentIntensityAnalyzer()
        self.last_retrain = datetime.utcnow() - timedelta(minutes=CONFIG['retrain_interval_minutes'] + 1)
        self._news_cache = {}
        self.running = False 

    def analisar_noticias(self, max_entries_per_feed=2):
        scores = []
        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_entries_per_feed]:
                    title = entry.title if hasattr(entry, 'title') else ''
                    if not title: continue
                    if title in self._news_cache: comp = self._news_cache[title]
                    else:
                        comp = self.analyzer.polarity_scores(title)['compound']
                        self._news_cache[title] = comp
                    scores.append(comp)
            except Exception: continue
        return float(np.mean(scores)) if scores else 0.0

    def buscar_dados_tecnicos(self, symbol, limit=1500, retry=3):
        tries = 0
        while tries < retry:
            try:
                time.sleep(CONFIG['sleep_before_ohlcv_call'])
                ohlcv = self.exchange.fetch_ohlcv(symbol, CONFIG['timeframe'], limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                df['rsi'] = df.ta.rsi(length=14)
                macd = df.ta.macd(fast=12, slow=26, signal=9)
                df['macd'] = macd.iloc[:,0] if (macd is not None and not macd.empty) else np.nan
                df['ema_50'] = df.ta.ema(length=50)
                df['atr'] = df.ta.atr(length=14)
                bb = df.ta.bbands(length=20, std=2)
                if bb is not None and not bb.empty:
                    df['bb_lower'] = bb.iloc[:,0]; df['bb_upper'] = bb.iloc[:,2]
                else: df['bb_lower'] = np.nan; df['bb_upper'] = np.nan
                df['obv'] = df.ta.obv()
                df['ret_1'] = df['close'].pct_change(1)
                df['ret_3'] = df['close'].pct_change(3)
                df['ret_6'] = df['close'].pct_change(6)
                df['ma_10'] = df['close'].rolling(10).mean()
                df['ma_20'] = df['close'].rolling(20).mean()
                df['ma_50'] = df['close'].rolling(50).mean()
                df['vol_rolling_std_20'] = df['close'].rolling(20).std()
                df['price_z'] = (df['close'] - df['ma_20'])/(df['vol_rolling_std_20']+1e-9)
                df['momentum_12'] = df['close']/df['close'].shift(12) - 1
                
                w = CONFIG['vwap_window']
                pv = (df['close'] * df['volume']).rolling(w).sum()
                vol_sum = df['volume'].rolling(w).sum()
                df['vwap'] = pv / vol_sum
                df['dist_vwap'] = (df['close'] - df['vwap'])/df['vwap']
                
                df.dropna(inplace=True)
                return df
            except Exception as e:
                tries += 1
                time.sleep(1.5)
        raise RuntimeError(f"Falha ao buscar OHLCV para {symbol}")

    def generate_triple_barrier_labels(self, df, atr_stop_mult=None, atr_take_mult=None, max_bars=None):
        df = df.copy().reset_index(drop=True)
        n = len(df)
        if max_bars is None: max_bars = CONFIG['lookahead_candles']
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            entry_price = df.at[i, 'close']
            atr = df.at[i, 'atr'] if 'atr' in df.columns else np.nan
            if pd.isna(atr) or atr <= 0 or atr_stop_mult is None: labels[i] = 0; continue
            stop_price = entry_price - (atr * atr_stop_mult)
            take_price = entry_price + (atr * atr_take_mult)
            hit = 0
            for j in range(i+1, min(n, i+1+max_bars)):
                if df.at[j, 'low'] <= stop_price: hit = 0; break
                if df.at[j, 'high'] >= take_price: hit = 1; break
            labels[i] = hit
        return labels

    def construir_target(self, df):
        look = CONFIG['lookahead_candles']
        df = df.copy()
        df['future_close'] = df['close'].shift(-look)
        df['future_return'] = (df['future_close'] / df['close']) - 1
        df.dropna(inplace=True)
        return df

    def treinar_modelo_para(self, symbol, override_quantile=None):
        logger.info(f"[TRAIN] 🧠 Treinando Inteligência para {symbol}... (Aguarde)")
        try:
            df = self.buscar_dados_tecnicos(symbol)
            if df.empty: return False
            
            labels = self.generate_triple_barrier_labels(df, atr_stop_mult=CONFIG['atr_stop_mult'], atr_take_mult=CONFIG['atr_take_mult'], max_bars=CONFIG['lookahead_candles'])
            df['target'] = labels
            
            features = [
                'rsi','macd','ema_50','atr','volume','bb_upper','bb_lower','obv',
                'ret_1','ret_3','ret_6','ma_10','ma_20','ma_50','vol_rolling_std_20',
                'price_z','momentum_12','dist_vwap'
            ]
            df_feat = df.dropna(subset=features + ['target']).copy()
            if len(df_feat) < 200: return False
            
            pos_rate = df_feat['target'].mean()
            if pos_rate < 0.02:
                tmp = self.construir_target(df)
                quant = override_quantile if override_quantile is not None else CONFIG['target_quantile']
                q_val = tmp['future_return'].quantile(quant)
                min_thresh = CONFIG['min_profit_margin'] + 2*CONFIG['fee_pct']
                threshold = max(q_val, min_thresh)
                tmp['target'] = (tmp['future_return'] > threshold).astype(int)
                df_feat = tmp.dropna(subset=features + ['target']).copy()
                
            X = df_feat[features]
            y = df_feat['target']
            if len(y.unique()) < 2: return False
            
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            split = int(len(X)*0.8)
            X_test, y_test = X.iloc[split:], y.iloc[split:]
            preds = rf.predict(X_test)
            oos_acc = accuracy_score(y_test, preds)
            
            logger.info(f"[TRAIN] ✅ {symbol} treinado: Acurácia Teste = {oos_acc:.2%}")
            
            meta = {'trained_at': now_str(), 'oos_acc': oos_acc, 'features': features}
            meta['ok_for_trading'] = True if oos_acc >= CONFIG['min_oos_accuracy'] else False
            
            joblib.dump({'model': rf, 'features': features}, model_path(symbol))
            with open(metadata_path(symbol), 'w') as f: json.dump(meta, f, indent=2)
            
            self.modelos[symbol] = {'model': rf, 'features': features, 'oos_acc': oos_acc, 'ok': meta['ok_for_trading']}
            save_feature_importances(symbol, features, rf.feature_importances_)
            return True
        except Exception as e:
            logger.error(f"[TRAIN ERR] {symbol}: {e}")
            return False

    def carregar_modelos_salvos(self):
        for sym in CONFIG['symbols']:
            p = model_path(sym)
            mpath = metadata_path(sym)
            if os.path.exists(p):
                try:
                    data = joblib.load(p)
                    meta = {}
                    if os.path.exists(mpath):
                        with open(mpath,'r') as f: meta = json.load(f)
                    self.modelos[sym] = {'model': data['model'], 'features': data['features'], 'oos_acc': meta.get('oos_acc', None), 'ok': meta.get('ok_for_trading', False)}
                    # logger.info(f"[LOAD] Modelo {sym} carregado (OK: {self.modelos[sym]['ok']})")
                except Exception: pass

    # --- MÉTODO TURBO COM ATUALIZAÇÃO DE PREÇOS ---
    def executar_ciclo_api(self):
        logger.info("--- ROBÔ INICIADO (MODO TURBO) ---")
        self.carregar_modelos_salvos()
        self.running = True
        
        while self.running:
            try:
                # Retreino (a cada 24h)
                if datetime.utcnow() - self.last_retrain > timedelta(minutes=CONFIG['retrain_interval_minutes']):
                    logger.info("[RETRAIN] ⏳ Iniciando ciclo de retreino programado...")
                    for sym in CONFIG['symbols']:
                        if not self.running: break
                        self.treinar_modelo_para(sym)
                    self.last_retrain = datetime.utcnow()

                # Sentimento
                sentimento = self.analisar_noticias()
                sent_norm = (sentimento + 1)/2
                logger.info(f"[STATUS] News: {sentimento:.2f} | Saldo: ${self.carteira.saldo:.2f}")

                # Posições
                for s in list(self.carteira.posicoes.keys()):
                    if not self.running: break
                    try:
                        tick = self.exchange.fetch_ticker(s)
                        last_known_prices[s] = tick['last']
                        lucro = self.carteira.calcular_lucro_flutuante(s, tick['last'])
                        self.carteira.verificar_saidas(s, tick['last'])
                    except Exception: pass

                # Scanner Rápido
                logger.info(f"[CYCLE] 🔄 Iniciando varredura de {len(CONFIG['symbols'])} ativos...")
                
                for sym in CONFIG['symbols']:
                    if not self.running: break
                    time.sleep(CONFIG['sleep_between_symbol_calls']) # 0.5s
                    
                    if sym in self.carteira.posicoes: 
                        continue
                    
                    try:
                        # Busca Preço Atualizado
                        df = self.buscar_dados_tecnicos(sym, limit=200)
                        price = df.iloc[-1]['close']
                        vwap = df.iloc[-1]['vwap']
                        atr = df.iloc[-1]['atr'] if 'atr' in df.columns else None
                        rsi = df.iloc[-1]['rsi']
                        
                        # Envia preço pro Front
                        last_known_prices[sym] = price

                        if sym not in self.modelos: continue
                        md = self.modelos.get(sym)
                        
                        if not md.get('ok', False) and CONFIG['enable_live_trading']:
                            continue
                            
                        features_data = df.iloc[[-1]][md.get('features')]
                        prob = md['model'].predict_proba(features_data)[0][1]
                        
                        prob_threshold_adj = 0.15 if sent_norm < 0.35 else 0.0
                        score = (prob * 0.7) + (sent_norm * 0.3) - prob_threshold_adj
                        
                        trend_filter = price > (vwap * CONFIG['vwap_trend_tolerance'])
                        
                        # ---- LOG DINÂMICO AQUI ----
                        trend_icon = "📈" if trend_filter else "📉"
                        logger.info(f"[SCAN] 🔎 {sym} | ${price:.2f} | RSI:{rsi:.0f} {trend_icon} | IA:{prob:.2f} | Score:{score:.2f}")
                        
                        # Lógica de Compra
                        if score > CONFIG['confianca_minima'] and prob > 0.50 and trend_filter:
                            self.carteira.comprar(sym, price, score, atr=atr)
                        else:
                            # Log Explicativo se for interessante
                            if score > 0.40: 
                                motivos = []
                                if score <= CONFIG['confianca_minima']: motivos.append(f"Score({score:.2f}) Baixo")
                                if prob <= 0.50: motivos.append(f"Prob.IA({prob:.2f}) Fraca")
                                if not trend_filter: motivos.append("Contra Tendência")
                                logger.info(f"[SKIP] ✋ {sym} | {', '.join(motivos)}")

                    except Exception as e:
                        logger.warning(f"[ERR] {sym}: {e}")

                # Espera curta
                logger.info("[WAIT] 💤 Aguardando próximo ciclo...")
                for _ in range(5): 
                    if not self.running: break
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Erro ciclo: {e}")
                time.sleep(5)
        
        logger.info("--- ROBÔ PARADO ---")

# -------------------------- API SETUP -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot_instance = RoboTrader(paper=True)
bot_thread = None

@app.get("/api/status")
def get_status():
    pnl_aberto = 0
    posicoes_list = []
    
    for sym, dados in bot_instance.carteira.posicoes.items():
        pnl = 0 
        if sym in last_known_prices:
            atual = last_known_prices[sym]
            pnl = (dados['quantidade'] * atual) - dados['valor_investido']
            
        posicoes_list.append({
            "symbol": sym,
            "entryPrice": dados['preco_entrada'],
            "quantity": dados['quantidade'],
            "invested": dados['valor_investido'],
            "pnl": pnl,
            "currentPrice": last_known_prices.get(sym, dados['preco_entrada'])
        })
        pnl_aberto += pnl

    total_investido = sum([p['valor_investido'] for p in bot_instance.carteira.posicoes.values()])
    equity = bot_instance.carteira.saldo + total_investido + pnl_aberto

    # Captura evento e limpa
    global last_trade_event
    event = last_trade_event
    last_trade_event = None 

    return {
        "isRunning": bot_instance.running,
        "balance": bot_instance.carteira.saldo,
        "equity": equity,
        "openPositions": posicoes_list,
        "dailyTrades": len(bot_instance.carteira.trades_today),
        "totalTrades": len(bot_instance.carteira.trades_fechados),
        "marketPrices": last_known_prices, # IMPORTANTE: Envia preços
        "lastEvent": event # IMPORTANTE: Envia notificação
    }

@app.get("/api/logs")
def get_logs(): return log_buffer

@app.get("/api/history")
def get_history(): return bot_instance.carteira.trades_fechados[-50:]

@app.post("/api/start")
def start_bot():
    global bot_thread
    if not bot_instance.running:
        bot_thread = threading.Thread(target=bot_instance.executar_ciclo_api)
        bot_thread.start()
        return {"status": "started"}
    return {"status": "already_running"}

@app.post("/api/stop")
def stop_bot():
    if bot_instance.running:
        bot_instance.running = False
        return {"status": "stopping"}
    return {"status": "not_running"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)