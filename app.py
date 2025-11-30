# app.py
"""
Versão aprimorada e CORRIGIDA.
Correções:
 - REMOVIDO o hack de 'WERKZEUG_RUN_MAIN' que causava o crash no Windows.
 - Corrigido DeprecationWarning do datetime (usando timezone-aware).
 - Mantida a supressão de logs HTTP.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import os
import sys
import threading
import time
from functools import wraps
from math import floor
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

import ccxt
import joblib
import pandas as pd
import pandas_ta as ta
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.serving import WSGIRequestHandler

# =========================
# 🔇 OPÇÃO NUCLEAR: SUPRESSÃO TOTAL DE LOGS
# =========================
# 1. Desativa logger padrão do Werkzeug
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
log.disabled = True 

# 2. O PULO DO GATO: Sobrescreve o método de log interno do servidor HTTP
# Isso impede que ele imprima "GET /api/status 200" no terminal
def log_request(self, code='-', size='-'):
    pass

WSGIRequestHandler.log_request = log_request
WSGIRequestHandler.log = lambda self, type, message, *args: None

# =========================
# CONFIG DEFAULTS + PATHS
# =========================
FORCAR_RESET_AGORA = False 

caminhos_possiveis = [
    r'G:\Meu Drive', r'G:\My Drive',
    r'D:\Meu Drive', r'E:\Meu Drive',
    os.path.expanduser('~/Google Drive'),
    os.getcwd()
]
BASE_DRIVE = next((p for p in caminhos_possiveis if os.path.exists(p)), os.getcwd())
PASTA_PROJETO = os.path.join(BASE_DRIVE, 'RoboTrader_Arquivos_final')
META_DIR = os.path.join(PASTA_PROJETO, 'model_metadata')
INTEL_DIR = os.path.join(PASTA_PROJETO, 'inteligencia_ia')
POSITIONS_FILE = os.path.join(PASTA_PROJETO, 'positions.json')
HISTORY_FILE = os.path.join(PASTA_PROJETO, 'history.json')
LIVE_STATE_FILE = os.path.join(PASTA_PROJETO, 'live_state.json')
METRICS_FILE = os.path.join(PASTA_PROJETO, 'metrics.json')
LOG_FILE = os.path.join(PASTA_PROJETO, 'bot_log.jsonl')

os.makedirs(META_DIR, exist_ok=True)
os.makedirs(INTEL_DIR, exist_ok=True)
os.makedirs(PASTA_PROJETO, exist_ok=True)

# =========================
# CONFIGURAÇÕES
# =========================
CONFIG = {
    'groups': {
        'large_cap': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        'mid_cap': ['SOL/USDT', 'ADA/USDT', 'MATIC/USDT', 'LINK/USDT'],
        'small_cap': ['PEPE/USDT', 'SHIB/USDT', 'DOGE/USDT']
    },
    'symbols': [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
        'AVAX/USDT', 'DOT/USDT', 'TRX/USDT', 'LINK/USDT', 'MATIC/USDT', 'ATOM/USDT',
        'NEAR/USDT', 'ALGO/USDT', 'ICP/USDT', 'FTM/USDT', 'APT/USDT', 'SUI/USDT',
        'ARB/USDT', 'OP/USDT', 'IMX/USDT', 'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT'
    ],
    'paper_trading': True,
    'initial_capital_brl': 100.0,
    'initial_capital_usdt': None, 
    'base_quote': 'USDT',
    'timeframe': '1h',
    'min_confidence': 0.55,
    'sleep_cycle': 60,
    'max_positions': 3,
    'stop_loss_mult': 2.0,
    'take_profit_mult': 5.0,
    'risk_per_trade': 0.015,
    'min_trade_usd': 5.0,
    'max_daily_trades': 6,
    'min_volume_quote': 5000.0,
    'slippage_pct': 0.0015,
    'fee_pct': 0.001,
    'max_exposure_pct': 0.30,
    'group_cooldown_seconds': 60 * 60,
    'circuit_breaker_failures': 8,
    'circuit_breaker_seconds': 10 * 60,
    'retry_max_attempts': 3,
    'retry_backoff_base': 1.0,
}

# Mapa auxiliar Símbolo -> Grupo
SYMBOL_TO_GROUP = {}
for g, syms in CONFIG['groups'].items():
    for s in syms:
        SYMBOL_TO_GROUP[s] = g
for s in CONFIG['symbols']:
    SYMBOL_TO_GROUP.setdefault(s, 'ungrouped')

# =========================
# LOGGER ESTRUTURADO (Visual Limpo)
# =========================
class StructuredLogger:
    def __init__(self, file_path: str = LOG_FILE):
        self.file_path = file_path
        self.buffer: List[Dict[str, Any]] = []
        self.limit = 1000
        
        # Configura logger apenas para INFO/ERRO do nosso sistema
        self._pylogger = logging.getLogger('robo_logger')
        self._pylogger.setLevel(logging.INFO)
        self._pylogger.propagate = False 
        
        if self._pylogger.hasHandlers():
            self._pylogger.handlers.clear()
            
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s')) 
        handler.setLevel(logging.INFO)
        self._pylogger.addHandler(handler)

    def _entry(self, level: str, message: str, **extra):
        # Correção do DeprecationWarning: usando timezone UTC explícito
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        ts = now_utc.strftime('%H:%M:%S')
        ts_iso = now_utc.isoformat().replace('+00:00', 'Z')
        
        entry = {
            "timestamp": ts_iso,
            "level": level,
            "message": message
        }
        entry.update(extra)
        self.buffer.append(entry)
        if len(self.buffer) > self.limit:
            self.buffer.pop(0)
            
        # Gravar em arquivo (backup)
        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        except:
            pass

        # Print no Terminal (Apenas o essencial)
        color_code = "0"
        if level == "INFO": color_code = "96"    # Ciano
        elif level == "SUCCESS": color_code = "92" # Verde
        elif level == "WARNING": color_code = "93" # Amarelo
        elif level == "ERROR": color_code = "91"   # Vermelho
        elif level == "SYSTEM": color_code = "95"  # Magenta

        extra_str = f" | {extra}" if extra else ""
        formatted_msg = f"\033[{color_code}m[{ts}] {level}: {message}{extra_str}\033[0m"
        
        self._pylogger.info(formatted_msg)

    def info(self, message: str, **extra): self._entry("INFO", message, **extra)
    def success(self, message: str, **extra): self._entry("SUCCESS", message, **extra)
    def warning(self, message: str, **extra): self._entry("WARNING", message, **extra)
    def error(self, message: str, **extra): self._entry("ERROR", message, **extra)
    def system(self, message: str, **extra): self._entry("SYSTEM", message, **extra)

logger = StructuredLogger()

# =========================
# UTILITÁRIOS (Circuit Breaker & Retry)
# =========================
class CircuitBreaker:
    def __init__(self, max_failures: int, reset_seconds: int):
        self.max_failures = max_failures
        self.reset_seconds = reset_seconds
        self.fail_count = 0
        self.lock = Lock()
        self.tripped_until: Optional[datetime.datetime] = None

    def record_success(self):
        with self.lock:
            self.fail_count = 0

    def record_failure(self):
        with self.lock:
            self.fail_count += 1
            if self.fail_count >= self.max_failures:
                self.tripped_until = datetime.datetime.now() + datetime.timedelta(seconds=self.reset_seconds)
                logger.warning("Circuit Breaker ATIVADO! Pausando operações.", until=self.tripped_until.isoformat())

    def allow(self) -> bool:
        with self.lock:
            if self.tripped_until is None:
                return True
            if datetime.datetime.now() >= self.tripped_until:
                self.tripped_until = None
                self.fail_count = 0
                logger.system("Circuit Breaker RESETADO. Retomando operações.")
                return True
            return False

circuit_breaker = CircuitBreaker(CONFIG['circuit_breaker_failures'], CONFIG['circuit_breaker_seconds'])

def retry_on_exception(max_attempts: int = 3, base_backoff: float = 1.0):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    if not circuit_breaker.allow():
                        return None 
                    result = func(*args, **kwargs)
                    circuit_breaker.record_success()
                    return result
                except Exception as e:
                    last_exc = e
                    circuit_breaker.record_failure()
                    if attempt < max_attempts:
                        time.sleep(base_backoff * (2 ** (attempt - 1)))
            logger.error(f"Falha na operação {func.__name__} após {max_attempts} tentativas.", error=str(last_exc))
            raise last_exc
        return wrapper
    return deco

# =========================
# CARTEIRA (Persistência + Reset Diário)
# =========================
class Carteira:
    def __init__(self, exchange):
        self.posicoes: Dict[str, Any] = {}
        self.historico: List[Dict[str, Any]] = []
        self.lock = Lock()
        self.last_event = None
        self.daily_trades = 0
        self.last_reset_day = datetime.date.today()
        self.exchange = exchange
        self.cash = 0.0
        self.load_or_init()

    def _convert_brl_to_usdt(self, brl_amount: float) -> float:
        try:
            t = self.exchange.fetch_ticker('USDT/BRL')
            rate = t.get('last') or t.get('close')
            if rate and rate > 0:
                usdt = brl_amount / rate
                return usdt
        except:
            pass
        return brl_amount

    def load_or_init(self):
        if CONFIG.get('initial_capital_usdt') is None:
            try:
                CONFIG['initial_capital_usdt'] = self._convert_brl_to_usdt(CONFIG['initial_capital_brl'])
            except:
                CONFIG['initial_capital_usdt'] = CONFIG['initial_capital_brl']
        
        self.cash = CONFIG['initial_capital_usdt']
        
        if FORCAR_RESET_AGORA:
            logger.warning("⚠️ RESET FORÇADO ATIVADO")
            self.posicoes = {}
            self.historico = []
            self.salvar()
            return
        
        # Carrega Posições
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, 'r') as f:
                    data = json.load(f)
                self.posicoes = data.get('posicoes', {})
                self.cash = float(data.get('cash', self.cash))
                self.daily_trades = int(data.get('daily_trades', 0))
                
                last_reset_str = data.get('last_reset_day')
                if last_reset_str:
                    self.last_reset_day = datetime.datetime.strptime(last_reset_str, "%Y-%m-%d").date()
                
                logger.info("Estado carregado", posicoes=len(self.posicoes), daily_trades=self.daily_trades)
            except Exception as e:
                logger.error("Erro ao carregar posições", error=str(e))
        
        # Carrega Histórico
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    self.historico = json.load(f)
            except:
                pass

    def salvar(self):
        try:
            with open(POSITIONS_FILE, 'w') as f:
                json.dump({
                    'posicoes': self.posicoes, 
                    'cash': self.cash,
                    'last_reset_day': self.last_reset_day.isoformat(),
                    'daily_trades': self.daily_trades
                }, f, indent=2, default=str)
            with open(HISTORY_FILE, 'w') as f:
                json.dump(self.historico, f, indent=2, default=str)
        except Exception as e:
            logger.error("Erro ao salvar dados", error=str(e))

    def _check_daily_reset(self):
        """Reseta contador de trades se o dia mudou"""
        today = datetime.date.today()
        if today > self.last_reset_day:
            logger.system(f"Novo dia detectado ({today}). Resetando contador de trades.")
            self.daily_trades = 0
            self.last_reset_day = today
            self.salvar()

    def abrir_posicao(self, symbol, entry_price, quantity, atr, stop_price=None, take_price=None):
        with self.lock:
            self._check_daily_reset() 
            
            cost = entry_price * quantity
            fee = cost * CONFIG['fee_pct']
            
            if cost + fee > self.cash:
                logger.warning("Saldo insuficiente", symbol=symbol)
                return False
                
            self.cash -= (cost + fee)
            self.posicoes[symbol] = {
                'entry_price': float(entry_price),
                'quantity': float(quantity),
                'invested': float(cost),
                'atr': float(atr),
                'stop_price': float(stop_price) if stop_price else None,
                'take_price': float(take_price) if take_price else None,
                'trail_stop': None,
                'timestamp': str(datetime.datetime.now()),
                'cooldown_until': None
            }
            
            self.last_event = {'symbol': symbol, 'type': 'BUY', 'price': entry_price, 'qty': quantity}
            self.daily_trades += 1
            
            self.salvar()
            logger.success("COMPRA Executada", symbol=symbol, price=entry_price, qty=quantity)
            return True

    def fechar_posicao(self, symbol, exit_price, reason="Sinal"):
        with self.lock:
            self._check_daily_reset()
            
            if symbol not in self.posicoes:
                return None
                
            pos = self.posicoes.pop(symbol)
            revenue = exit_price * pos['quantity']
            fee = revenue * CONFIG['fee_pct']
            pnl = revenue - pos['invested'] - fee
            
            self.cash += (revenue - fee)
            
            record = {
                'symbol': symbol,
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'quantity': pos['quantity'],
                'pnl': pnl,
                'reason': reason,
                'timestamp': str(datetime.datetime.now())
            }
            
            self.historico.append(record)
            self.last_event = {'symbol': symbol, 'type': 'SELL', 'pnl': pnl, 'reason': reason}
            self.daily_trades += 1
            
            self.salvar()
            
            if pnl > 0:
                logger.success("VENDA (Lucro)", symbol=symbol, pnl=f"${pnl:.2f}", reason=reason)
            else:
                logger.warning("VENDA (Prejuízo)", symbol=symbol, pnl=f"${pnl:.2f}", reason=reason)
                
            return record

# =========================
# METRICS (Simples)
# =========================
class Metrics:
    def __init__(self):
        self.path = METRICS_FILE
        self.lock = Lock()

    def record_trade(self, pnl):
        with self.lock:
            if os.path.exists(self.path):
                try:
                    with open(self.path, 'r') as f: data = json.load(f)
                except: data = {}
            else:
                data = {}
            
            today = datetime.date.today().isoformat()
            if today not in data:
                data[today] = {'trades': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0}
                
            data[today]['trades'] += 1
            data[today]['pnl'] += pnl
            if pnl > 0: data[today]['wins'] += 1
            else: data[today]['losses'] += 1
            
            try:
                with open(self.path, 'w') as f: json.dump(data, f, indent=2)
            except: pass

metrics = Metrics()

# =========================
# ROBOTRADER (Cérebro)
# =========================
class RoboTrader:
    def __init__(self):
        self.running = False
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.carteira = Carteira(self.exchange)
        self.models = {}
        self.market_prices = {}
        self.cooldowns = {}
        self.group_cooldowns = {}
        self.live_state = self._load_live_state()
        
        # Histórico de Equity em Memória (para gráfico)
        self.equity_history = [] 
        
        if self.live_state.get("armed"):
            CONFIG['paper_trading'] = False
            
        self.load_models()
        logger.system("RoboTrader Pronto", paper_trading=CONFIG['paper_trading'])

    def _load_live_state(self):
        if os.path.exists(LIVE_STATE_FILE):
            try:
                with open(LIVE_STATE_FILE, 'r') as f: return json.load(f)
            except: pass
        return {"armed": False}

    def save_live_state(self, state):
        try:
            with open(LIVE_STATE_FILE, 'w') as f: json.dump(state, f)
            self.live_state = state
        except: pass

    def load_models(self):
        count = 0
        for sym in CONFIG['symbols']:
            safe_sym = sym.replace('/', '_')
            model_path = os.path.join(INTEL_DIR, f"{safe_sym}.joblib")
            meta_path = os.path.join(META_DIR, f"{safe_sym}_meta.json")
            if os.path.exists(model_path) and os.path.exists(meta_path):
                try:
                    self.models[sym] = {
                        'model': joblib.load(model_path),
                        'features': json.load(open(meta_path)).get('features', [])
                    }
                    count += 1
                except: pass
        logger.info(f"{count} Modelos de IA carregados.")

    @retry_on_exception(max_attempts=3)
    def safe_fetch_data(self, symbol):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=CONFIG['timeframe'], limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        return df

    @retry_on_exception(max_attempts=3)
    def safe_ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    def analyze(self, symbol):
        try:
            df = self.safe_fetch_data(symbol)
            if df is None or df.empty: return None
            
            df['c'] = df['c'].astype(float)
            current_close = df['c'].iloc[-1]
            self.market_prices[symbol] = current_close
            
            df['rsi'] = ta.rsi(df['c'], length=14)
            df['ema50'] = ta.ema(df['c'], length=50)
            df['atr'] = ta.atr(df['h'], df['l'], df['c'], length=14)
            
            row = df.iloc[-1].to_dict()
            return row
        except:
            return None

    def cycle(self):
        # Log limpo de ciclo
        logger.info(f"--- Ciclo Iniciado: {len(self.carteira.posicoes)} Posições Abertas ---")
        
        # 1. Update Patrimônio & Histórico para Gráfico
        invested_val = sum([p['invested'] for p in self.carteira.posicoes.values()])
        total_equity = self.carteira.cash + invested_val
        
        # Adiciona ponto no gráfico dinâmico
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        ts_now = now_utc.strftime('%H:%M:%S') # ou now_utc.astimezone() para local se preferir
        
        self.equity_history.append({'time': ts_now, 'value': total_equity})
        # Mantém apenas os últimos 50 pontos para não pesar
        if len(self.equity_history) > 50: self.equity_history.pop(0)

        # 2. Gerenciar Posições Abertas
        for sym in list(self.carteira.posicoes.keys()):
            try:
                tk = self.safe_ticker(sym)
                price = tk['last']
                self.market_prices[sym] = price
                pos = self.carteira.posicoes[sym]
                
                # Trailing Stop Simples
                if pos.get('atr'):
                    new_trail = price - pos['atr']
                    if pos.get('trail_stop') is None or new_trail > pos['trail_stop']:
                        pos['trail_stop'] = new_trail
                
                reason = None
                if pos.get('stop_price') and price <= pos['stop_price']: reason = "STOP LOSS"
                elif pos.get('take_price') and price >= pos['take_price']: reason = "TAKE PROFIT"
                elif pos.get('trail_stop') and price <= pos['trail_stop']: reason = "TRAILING STOP"
                
                if reason:
                    self.carteira.fechar_posicao(sym, price, reason=reason)
                    metrics.record_trade(0) 
            except:
                pass

        # 3. Novas Entradas
        if self.carteira.daily_trades >= CONFIG['max_daily_trades']:
            logger.info("Limite diário de trades atingido. Aguardando amanhã.")
            return

        slots = CONFIG['max_positions'] - len(self.carteira.posicoes)
        if slots <= 0: return

        candidates = []
        for sym in CONFIG['symbols']:
            if sym in self.carteira.posicoes: continue
            
            if self.cooldowns.get(sym) and datetime.datetime.now() < self.cooldowns[sym]: continue
            
            row = self.analyze(sym)
            if not row: continue
            
            # Score (IA ou Fallback)
            score = 0.5
            if row['rsi'] < 35: score += 0.2
            if row['c'] > row['ema50']: score += 0.1
            
            if score > CONFIG['min_confidence']:
                candidates.append({'symbol': sym, 'score': score, 'price': row['c'], 'atr': row['atr']})
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        for cand in candidates:
            if slots <= 0: break
            
            risk = total_equity * CONFIG['risk_per_trade']
            stop_dist = cand['atr'] * CONFIG['stop_loss_mult']
            qty = risk / stop_dist if stop_dist > 0 else 0
            
            cost = qty * cand['price']
            if cost < CONFIG['min_trade_usd']: continue
            if cost > self.carteira.cash: continue
            
            ok = self.carteira.abrir_posicao(
                cand['symbol'], cand['price'], qty, cand['atr'],
                stop_price=cand['price'] - stop_dist,
                take_price=cand['price'] + (cand['atr'] * CONFIG['take_profit_mult'])
            )
            
            if ok:
                slots -= 1
                self.cooldowns[cand['symbol']] = datetime.datetime.now() + datetime.timedelta(seconds=CONFIG['group_cooldown_seconds'])

    def loop(self):
        self.running = True
        while self.running:
            try: self.cycle()
            except Exception as e: logger.error(f"Erro no ciclo: {e}")
            
            for _ in range(CONFIG['sleep_cycle']):
                if not self.running: break
                time.sleep(1)

    def start(self):
        if not self.running:
            Thread(target=self.loop, daemon=True).start()
            logger.system("Robô Iniciado")

    def stop(self):
        self.running = False
        logger.system("Robô Parado")

# =========================
# API FLASK
# =========================
app = Flask(__name__)
CORS(app)
bot = RoboTrader()

@app.route('/api/start', methods=['POST'])
def api_start():
    bot.start()
    return jsonify({"status": "started"})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    bot.stop()
    return jsonify({"status": "stopped"})

@app.route('/api/logs', methods=['GET'])
def api_logs():
    # Retorna APENAS logs relevantes e coloridos do nosso buffer, sem lixo do sistema
    return jsonify(list(reversed(logger.buffer)))

@app.route('/api/history', methods=['GET'])
def api_history():
    return jsonify(bot.carteira.historico)

@app.route('/api/status', methods=['GET'])
def api_status():
    open_data = []
    invested_total = 0
    for sym, val in bot.carteira.posicoes.items():
        curr = bot.market_prices.get(sym, val['entry_price'])
        val_now = val['quantity'] * curr
        pnl = val_now - val['invested']
        open_data.append({
            "symbol": sym, "entryPrice": val['entry_price'],
            "quantity": val['quantity'], "invested": val['invested'],
            "pnl": pnl, "currentPrice": curr
        })
        invested_total += val['invested']
    
    eq = bot.carteira.cash + sum([p['invested'] + p['pnl'] for p in open_data])
    
    return jsonify({
        "isRunning": bot.running,
        "balance": bot.carteira.cash,
        "equity": eq,
        "equityHistory": bot.equity_history, # <-- ISTO alimenta o gráfico
        "dailyTrades": bot.carteira.daily_trades, # <-- ISTO alimenta "Trades Hoje"
        "totalTrades": len(bot.carteira.historico),
        "marketPrices": bot.market_prices,
        "lastEvent": bot.carteira.last_event,
        "openPositions": open_data
    })

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({"status": "ok", "uptime": "running" if bot.running else "stopped"})

if __name__ == '__main__':
    print("\033[92m=== SISTEMA DE TRADING INICIADO (MODO SILENCIOSO) ===\033[0m")
    print(f"\033[96mServidor rodando em http://localhost:5000\033[0m")
    app.run(host='0.0.0.0', port=5000)