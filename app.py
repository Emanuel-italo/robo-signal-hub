import os
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import json
import time
import datetime
import logging
from threading import Thread, Lock
from flask import Flask, jsonify, request
from flask_cors import CORS

# =============================================
# ⚠️ CONFIGURAÇÃO DE RESET (NUCLEAR)
# =============================================
FORCAR_RESET_AGORA = False  # Mude para True apenas se quiser zerar tudo

# =============================================
# CONFIGURAÇÕES GERAIS
# =============================================

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# === CONFIGURAÇÃO DE PASTA (GOOGLE DRIVE) ===
caminhos_possiveis = [
    r'G:\Meu Drive', r'G:\My Drive', 
    r'D:\Meu Drive', r'E:\Meu Drive',
    os.path.expanduser('~/Google Drive'),
    os.getcwd()
]
BASE_DRIVE = next((path for path in caminhos_possiveis if os.path.exists(path)), os.getcwd())
PASTA_PROJETO = os.path.join(BASE_DRIVE, 'RoboTrader_Arquivos_final')
META_DIR = os.path.join(PASTA_PROJETO, 'model_metadata')
INTEL_DIR = os.path.join(PASTA_PROJETO, 'inteligencia_ia')
POSITIONS_FILE = os.path.join(PASTA_PROJETO, 'positions.json')
HISTORY_FILE = os.path.join(PASTA_PROJETO, 'history.json') # Novo arquivo para histórico

os.makedirs(META_DIR, exist_ok=True)
os.makedirs(INTEL_DIR, exist_ok=True)

CONFIG = {
    'symbols': [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
        'AVAX/USDT', 'DOT/USDT', 'TRX/USDT', 'LINK/USDT', 'MATIC/USDT', 'ATOM/USDT',
        'NEAR/USDT', 'ALGO/USDT', 'ICP/USDT', 'FTM/USDT', 'APT/USDT', 'SUI/USDT',
        'ARB/USDT', 'OP/USDT', 'IMX/USDT',
        'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT',
        'UNI/USDT', 'AAVE/USDT', 'LTC/USDT', 'BCH/USDT', 'ETC/USDT', 'FIL/USDT',
        'RNDR/USDT', 'FET/USDT', 'GRT/USDT', 'SAND/USDT', 'MANA/USDT', 'AXS/USDT',
        'GALA/USDT'
    ],
    'paper_trading': True,        
    'initial_capital': 100.0,     
    'timeframe': '1h',
    'min_confidence': 0.55,       
    'sleep_cycle': 60,            
    'max_positions': 3,           
    'stop_loss_mult': 2.0,        
    'take_profit_mult': 5.0       
}

# =============================================
# SISTEMA DE LOGS
# =============================================

class FrontendLogger:
    def __init__(self):
        self.buffer = []
        self.limit = 100

    def _add(self, level, msg):
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {level}: {msg}")
        log_entry = {"time": ts, "level": level, "message": msg}
        self.buffer.append(log_entry)
        if len(self.buffer) > self.limit:
            self.buffer.pop(0)

    def info(self, msg): self._add("INFO", msg)
    def success(self, msg): self._add("SUCCESS", msg)
    def warning(self, msg): self._add("WARNING", msg)
    def error(self, msg): self._add("ERROR", msg)
    def system(self, msg): self._add("SYSTEM", msg)

logger = FrontendLogger()

# =============================================
# CARTEIRA
# =============================================

class Carteira:
    def __init__(self):
        self.posicoes = {}
        self.historico = [] # Novo: Guarda histórico de trades fechados
        self.cash = CONFIG['initial_capital']
        self.lock = Lock()
        
        self.last_event = None # Novo: Para notificação no frontend
        self.daily_trades = 0  # Novo: Contador diário
        
        if FORCAR_RESET_AGORA:
            logger.warning("⚠️ RESET FORÇADO ATIVADO! Apagando memória antiga...")
            self.posicoes = {}
            self.historico = []
            self.cash = CONFIG['initial_capital']
            self.salvar()
            logger.success(f"Sistema restaurado para R$ {self.cash:.2f}")
        else:
            self.carregar()

    def carregar(self):
        # Carrega Posições
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, 'r') as f:
                    data = json.load(f)
                    self.posicoes = data.get('posicoes', {})
                    self.cash = float(data.get('cash', CONFIG['initial_capital']))
                logger.info(f"Memória carregada. Saldo Atual: {self.cash:.2f}")
            except Exception as e:
                logger.error(f"Erro ao carregar posições: {e}")
        
        # Carrega Histórico
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    self.historico = json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar histórico: {e}")

    def salvar(self):
        try:
            with open(POSITIONS_FILE, 'w') as f:
                json.dump({'posicoes': self.posicoes, 'cash': self.cash}, f, indent=2)
            
            with open(HISTORY_FILE, 'w') as f:
                json.dump(self.historico, f, indent=2)
        except Exception as e:
            logger.error(f"Erro ao salvar: {e}")

    def abrir_posicao(self, symbol, price, quantity, atr):
        with self.lock:
            cost = price * quantity
            if cost > self.cash:
                return False
            
            self.cash -= cost
            self.posicoes[symbol] = {
                'entry_price': price,
                'quantity': quantity,
                'invested': cost,
                'atr': atr,
                'timestamp': str(datetime.datetime.now())
            }
            
            # Evento para Frontend
            self.last_event = {
                'symbol': symbol,
                'type': 'BUY',
                'price': price
            }
            self.daily_trades += 1
            
            self.salvar()
            logger.success(f"COMPRA: {symbol} | Investido: ${cost:.2f}")
            return True

    def fechar_posicao(self, symbol, price, reason="Sinal"):
        with self.lock:
            if symbol not in self.posicoes: return
            
            pos = self.posicoes.pop(symbol)
            revenue = pos['quantity'] * price
            profit = revenue - pos['invested']
            self.cash += revenue
            
            # Registra no Histórico
            trade_record = {
                'symbol': symbol,
                'entry_price': pos['entry_price'],
                'exit_price': price,
                'pnl': profit,
                'reason': reason,
                'timestamp': str(datetime.datetime.now())
            }
            self.historico.append(trade_record)
            
            # Evento para Frontend
            self.last_event = {
                'symbol': symbol,
                'type': 'SELL',
                'pnl': profit
            }
            self.daily_trades += 1
            
            self.salvar()
            
            cor = "SUCCESS" if profit > 0 else "WARNING"
            self._log_custom(cor, f"VENDA {symbol}: {reason} | PnL: ${profit:.2f} | Saldo: {self.cash:.2f}")

    def _log_custom(self, level, msg):
        if level == "SUCCESS": logger.success(msg)
        else: logger.warning(msg)

# =============================================
# CÉREBRO
# =============================================

class RoboTrader:
    def __init__(self):
        self.running = False
        self.carteira = Carteira()
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.market_prices = {} # Novo: Armazena preços para o ticker
        self.load_models()
        logger.system(f"Estratégia: Dividir R$ {CONFIG['initial_capital']} em {CONFIG['max_positions']} posições.")

    def load_models(self):
        loaded = 0
        for sym in CONFIG['symbols']:
            safe_sym = sym.replace('/', '_')
            model_path = os.path.join(INTEL_DIR, f"{safe_sym}.joblib")
            meta_path = os.path.join(META_DIR, f"{safe_sym}_meta.json")

            if os.path.exists(model_path) and os.path.exists(meta_path):
                try:
                    model = joblib.load(model_path)
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    self.models[sym] = {'model': model, 'features': meta['features']}
                    loaded += 1
                except Exception as e:
                    logger.error(f"Erro IA {sym}: {e}")
        logger.info(f"{loaded} IAs prontas.")

    def get_data_and_features(self, symbol):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=CONFIG['timeframe'], limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Atualiza preço de mercado global
            current_close = df['close'].iloc[-1]
            self.market_prices[symbol] = current_close
            
            df['rsi'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'])
            macd_col = [c for c in macd.columns if c.startswith('MACD_') and not c.startswith('MACDs') and not c.startswith('MACDh')]
            df['macd'] = macd[macd_col[0]] if macd_col else 0
            
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            bb = ta.bbands(df['close'], length=20, std=2)
            bbu_col = [c for c in bb.columns if c.startswith('BBU')]
            df['bb_upper'] = bb[bbu_col[0]] if bbu_col else df['close']

            df['dist_ema'] = (df['close'] - df['ema_50']) / df['ema_50']
            df['dist_bb_up'] = (df.get('bb_upper', df['close']) - df['close']) / df['close']
            df['rsi_norm'] = df['rsi'] / 100
            df['vol_change'] = df['volume'].pct_change()
            
            return df.iloc[-1], df
        except:
            return None, None

    def cycle(self):
        logger.info(f"--- Varredura (Vagas: {CONFIG['max_positions'] - len(self.carteira.posicoes)}) ---")
        
        # 1. Gerencia Saídas e Atualiza Preços das Posições
        for sym in list(self.carteira.posicoes.keys()):
            try:
                ticker = self.exchange.fetch_ticker(sym)
                current_price = ticker['last']
                
                # Atualiza preço global
                self.market_prices[sym] = current_price
                
                pos = self.carteira.posicoes[sym]
                
                stop = pos['entry_price'] - (pos['atr'] * CONFIG['stop_loss_mult'])
                take = pos['entry_price'] + (pos['atr'] * CONFIG['take_profit_mult'])
                
                if current_price <= stop: self.carteira.fechar_posicao(sym, current_price, "STOP LOSS")
                elif current_price >= take: self.carteira.fechar_posicao(sym, current_price, "TAKE PROFIT")
            except: pass

        # 2. Verifica Vagas
        slots_abertos = CONFIG['max_positions'] - len(self.carteira.posicoes)
        if slots_abertos <= 0:
            logger.info("Carteira Completa. Monitorando saídas...")
            return

        # 3. Busca Oportunidades
        candidates = []
        for sym in CONFIG['symbols']:
            if sym in self.carteira.posicoes or sym not in self.models: continue
            
            time.sleep(0.5) 
            row, _ = self.get_data_and_features(sym)
            if row is None: continue
            
            try:
                model_data = self.models[sym]
                input_df = pd.DataFrame([row[model_data['features']]]).fillna(0)
                prob = model_data['model'].predict_proba(input_df)[0][1]
                
                if prob > CONFIG['min_confidence']:
                    candidates.append({
                        'symbol': sym, 'prob': prob, 
                        'price': row['close'], 'atr': row['atr']
                    })
            except: pass

        # 4. Executa Compras
        candidates.sort(key=lambda x: x['prob'], reverse=True)

        for cand in candidates:
            if slots_abertos <= 0: break
            
            if self.carteira.cash < 5.0:
                logger.warning("Saldo insuficiente para nova posição.")
                break

            if slots_abertos == 1:
                invest_amount = self.carteira.cash
            else:
                invest_amount = self.carteira.cash / slots_abertos

            invest_amount = invest_amount * 0.99

            if invest_amount > 5.0:
                qty = invest_amount / cand['price']
                if self.carteira.abrir_posicao(cand['symbol'], cand['price'], qty, cand['atr']):
                    slots_abertos -= 1
            else:
                logger.warning(f"Quantia {invest_amount:.2f} muito baixa para operar {cand['symbol']}")

    def loop(self):
        self.running = True
        while self.running:
            try: self.cycle()
            except Exception as e: logger.error(f"Erro Loop: {e}")
            for _ in range(CONFIG['sleep_cycle']):
                if not self.running: break
                time.sleep(1)

    def start(self): 
        if not self.running: Thread(target=self.loop, daemon=True).start()
    def stop(self): self.running = False

# =============================================
# API
# =============================================
app = Flask(__name__)
CORS(app)
bot = RoboTrader()

@app.route('/api/start', methods=['POST'])
def start(): 
    bot.start()
    return jsonify({"status": "started"})

@app.route('/api/stop', methods=['POST'])
def stop(): 
    bot.stop()
    return jsonify({"status": "stopped"})

@app.route('/api/logs', methods=['GET'])
def logs(): return jsonify(list(reversed(logger.buffer)))

# --- NOVA ROTA: Histórico para o Gráfico de Win Rate ---
@app.route('/api/history', methods=['GET'])
def history():
    return jsonify(bot.carteira.historico)

@app.route('/api/status', methods=['GET'])
def status():
    # Calcula PnL não realizado para cada posição aberta
    open_pos_data = []
    invested_total = 0
    
    for sym, val in bot.carteira.posicoes.items():
        # Pega preço atual se disponível, senão usa preço de entrada
        curr_price = bot.market_prices.get(sym, val['entry_price'])
        # Valor atual da posição
        curr_val = val['quantity'] * curr_price
        # PnL não realizado
        unrealized_pnl = curr_val - val['invested']
        
        open_pos_data.append({
            "symbol": sym, 
            "entryPrice": val['entry_price'], 
            "quantity": val['quantity'], 
            "invested": val['invested'], 
            "pnl": unrealized_pnl,
            "currentPrice": curr_price
        })
        invested_total += val['invested']

    equity = bot.carteira.cash + sum([(p['invested'] + p['pnl']) for p in open_pos_data])

    return jsonify({
        "isRunning": bot.running,
        "balance": bot.carteira.cash,
        "equity": equity,
        "dailyTrades": bot.carteira.daily_trades, # Adicionado
        "totalTrades": len(bot.carteira.historico), # Adicionado
        "marketPrices": bot.market_prices, # Adicionado para o Ticker
        "lastEvent": bot.carteira.last_event, # Adicionado para Toast
        "openPositions": open_pos_data
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)