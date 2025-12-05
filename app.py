# -*- coding: utf-8 -*-
"""
VERSÃO DE PRODUÇÃO (LIVE TRADING) - UPGRADE OPÇÃO C (COM CORREÇÕES + COMPRA MANUAL + CORREÇÃO NOTIONAL)
Corrige:
 - erro de precisão/min_amount ao vender (ex: DOGE)
 - erro de NOTIONAL (Filter failure -1013) para vendas pequenas (< 5 USD)
 - recuperação / exposição do histórico que sumiu
 - Lógica de diversificação forçada (Max Position %)
 - Adiciona Endpoint de Compra Manual
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import os
import sys
import time
from functools import wraps
from math import floor
from threading import Lock, Thread
from typing import Any, Dict, List, Optional
from decimal import Decimal, ROUND_DOWN

# BIBLIOTECAS EXTERNAS
import ccxt
import joblib
import pandas as pd
import pandas_ta as ta
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.serving import WSGIRequestHandler
from dotenv import load_dotenv  # Carrega as variáveis de ambiente

# Optional sklearn for partial_fit online updates (se disponível)
try:
    from sklearn.linear_model import SGDClassifier
    import numpy as np
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Carrega chaves do arquivo .env
load_dotenv()

# =========================
# 🔇 OPÇÃO NUCLEAR: SUPRESSÃO TOTAL DE LOGS DO FLASK/WSGI (FRONT-INTEGRATION PRESERVADA)
# =========================
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
log.disabled = True


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
SAMPLES_DIR = os.path.join(INTEL_DIR, 'samples')
POSITIONS_FILE = os.path.join(PASTA_PROJETO, 'positions.json')
HISTORY_FILE = os.path.join(PASTA_PROJETO, 'history.json')
LIVE_STATE_FILE = os.path.join(PASTA_PROJETO, 'live_state.json')
METRICS_FILE = os.path.join(PASTA_PROJETO, 'metrics.json')
LOG_FILE = os.path.join(PASTA_PROJETO, 'bot_log.jsonl')

os.makedirs(META_DIR, exist_ok=True)
os.makedirs(INTEL_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(PASTA_PROJETO, exist_ok=True)

# =========================
# CONFIGURAÇÕES (OTIMIZADAS PARA R$ 100 / 17 USDT) + OPÇÃO C
# =========================
CONFIG = {
    'groups': {
        'large_cap': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        'mid_cap': ['SOL/USDT', 'ADA/USDT', 'MATIC/USDT', 'LINK/USDT'],
        'small_cap': ['PEPE/USDT', 'SHIB/USDT', 'DOGE/USDT']
    },
    'symbols': [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
        'AVAX/USDT', 'DOT/USDT', 'TRX/USDT', 'LINK/USDT', 'ATOM/USDT', # REMOVIDO: MATIC
        'NEAR/USDT', 'ALGO/USDT', 'ICP/USDT', 'APT/USDT', 'SUI/USDT',  # REMOVIDO: FTM
        'ARB/USDT', 'OP/USDT', 'IMX/USDT', 'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT',
        'POL/USDT'
    ],
    # --- MODO REAL ---
    'paper_trading': False,  # <--- AGORA É REAL (mude para True para testes)
    'initial_capital_brl': 100.0,  # (Apenas referência, ele vai ler o saldo da Binance)
    'initial_capital_usdt': None,
    'base_quote': 'USDT',

    # --- ESTRATÉGIA ---
    'timeframe': '1h',
    'min_confidence': 0.55,
    'sleep_cycle': 60,

    # --- GERENCIAMENTO DE RISCO (AJUSTADO PARA BANCA PEQUENA) ---
    'max_positions': 5,        # Quantas posições simultâneas
    # Stop Loss desativado na lógica (ver abaixo), mantido aqui apenas config legado
    'stop_loss_mult': 2.0,   # Valor alto para garantir que não acione por acidente
    'take_profit_mult': 5.0,
    'risk_per_trade': 0.02,    # 2% do patrimônio por trade (ajustável)
    'min_trade_usd': 6.0,      # mínimo da Binance (deixe > 5)
    'max_daily_trades': 25,
    'min_volume_quote': 5000.0,
    'slippage_pct': 0.0015,
    'fee_pct': 0.001,
    'max_exposure_pct': 0.90,  # Permite usar 90% do saldo
    'group_cooldown_seconds': 60 * 60,

    # Diversificação / Alocação
    'diversify': True,
    'allocation_mode': 'proportional',  # 'proportional' ou 'equal'
    'allocation_buffer_pct': 0.10,  # Mantém 10% do cash como buffer para fees/ordens
    
    # --- NOVO: LIMITAR PORCENTAGEM DA BANCA EM UMA ÚNICA MOEDA ---
    'max_position_pct': 0.30,       # Max 30% do capital total em uma única moeda
    'min_allocation_per_slot': 6.5,  # mesmo que min_trade_usd (segurança)

    # comportamento automático para "dust" (quantias menores que min_amount da exchange)
    'auto_clean_dust': True,  # se True: limpa posições locais com quantia < min_amount (evita loop)

    # MODELO / LEARNING
    'online_learning': True,            # tenta partial_fit quando possível
    'retrain_every_n_trades': 50,       # quando atingir, tente re-treinar/exportar (não automático aqui)
    'min_expected_ev': 0.0005,         # min EV por USD de posição para considerar (ex.: 0.0005 USD por 1 USD investido)
    'conservative_mode': True,         # se True, enrijece stops/taking e reduz risco
    'max_drawdown_pct': 0.20,          # se drawdown > 20% pausa entradas

    # VENDA AO ALCANÇAR LUCRO (LOCK PROFIT)
    'min_profit_usd': 0.50,   # padrão conservador (ajuste conforme sua preferência)
    'min_profit_pct': 0.015,  # 0.5% do capital investido na posição

    # Logging
    'verbose_logs': False,  # se True mostra muitos logs; se False, apenas logs relevantes

    # --- SEGURANÇA ---
    'circuit_breaker_failures': 8,
    'circuit_breaker_seconds': 10 * 60,
    'retry_max_attempts': 3,
    'retry_backoff_base': 1.0,

    # --- OPÇÃO C (NOVOS PARÂMETROS) ---
    # Entrada escalonada (pyramiding) - frações que somam 1.0
    'entry_steps': [1.0],
    # espaçamento entre steps em múltiplos de ATR (ex.: primeiro step já, 2nd = -1.5 ATR, 3rd = -2.5 ATR)
    'entry_step_spacing_atr': [0.0],
    # trailing stop multiplier baseado em ATR
    'trailing_atr_multiplier': 1.0,
    # break-even: depois de X * ATR de lucro, move stop para entry (ou entry + tiny buffer)
    'break_even_profit_atr': 1.0,
    # take profit tiers adicionais (mantidos como Fallback se quiser modificar)
    'take_profit_tiers': [],
    # Média móvel adaptativa (KAMA) params
    'adaptive_ma': {'n': 10, 'fast': 2, 'slow': 30},

    # Modelo e threshold
    'model_prob_threshold': 0.55
}

# Mapa auxiliar Símbolo -> Grupo
SYMBOL_TO_GROUP = {}
for g, syms in CONFIG['groups'].items():
    for s in syms:
        SYMBOL_TO_GROUP[s] = g
for s in CONFIG['symbols']:
    SYMBOL_TO_GROUP.setdefault(s, 'ungrouped')

# =========================
# LOGGER ESTRUTURADO (APENAS LOGS RELEVANTES POR PADRÃO)
# =========================
class StructuredLogger:
    def __init__(self, file_path: str = LOG_FILE):
        self.file_path = file_path
        self.buffer: List[Dict[str, Any]] = []
        self.limit = 2000

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
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        ts_iso = now_utc.isoformat().replace('+00:00', 'Z')

        entry = {
            "timestamp": ts_iso,
            "level": level,
            "message": message
        }
        entry.update(extra)
        # Mantém buffer para frontend (API /api/logs)
        self.buffer.append(entry)
        if len(self.buffer) > self.limit:
            self.buffer.pop(0)

        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        except Exception:
            pass

        color_code = "0"
        if level == "INFO":
            color_code = "96"    # Ciano
        elif level == "SUCCESS":
            color_code = "92"  # Verde
        elif level == "WARNING":
            color_code = "93"  # Amarelo
        elif level == "ERROR":
            color_code = "91"   # Vermelho
        elif level == "SYSTEM":
            color_code = "95"  # Magenta

        # Remove dados sensíveis do print
        extra_clean = extra.copy()
        if 'api_key' in extra_clean:
            del extra_clean['api_key']

        extra_str = f" | {extra_clean}" if extra_clean else ""
        formatted_msg = f"\033[{color_code}m[{now_utc.strftime('%H:%M:%S')}] {level}: {message}{extra_str}\033[0m"

        self._pylogger.info(formatted_msg)

    def info(self, message: str, **extra):
        self._entry("INFO", message, **extra)

    def success(self, message: str, **extra):
        self._entry("SUCCESS", message, **extra)

    def warning(self, message: str, **extra):
        self._entry("WARNING", message, **extra)

    def error(self, message: str, **extra):
        self._entry("ERROR", message, **extra)

    def system(self, message: str, **extra):
        self._entry("SYSTEM", message, **extra)


logger = StructuredLogger()

# =========================
# UTILITÁRIOS
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
                logger.warning("Circuit Breaker ATIVADO! Pausando operações.")

    def allow(self) -> bool:
        with self.lock:
            if self.tripped_until is None:
                return True
            if datetime.datetime.now() >= self.tripped_until:
                self.tripped_until = None
                self.fail_count = 0
                logger.system("Circuit Breaker RESETADO.")
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
            logger.error(f"Falha na operação {func.__name__}", error=str(last_exc))
            return None  # Retorna None em vez de crashar
        return wrapper
    return deco


# ----- Técnicos: ATR e KAMA -----
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['h']
    low = df['l']
    close = df['c']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr


def kaufman_adaptive_moving_average(close: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    close = close.copy().reset_index(drop=True)
    n = int(n)
    fastSC = 2 / (fast + 1)
    slowSC = 2 / (slow + 1)
    kama = pd.Series(index=close.index, dtype=float)
    # inicializa com simples moving average nas primeiras janelas
    for i in range(len(close)):
        if i < n:
            kama.iloc[i] = close.iloc[i]
        else:
            change = abs(close.iloc[i] - close.iloc[i - n])
            volatility = close.iloc[i - n + 1:i + 1].diff().abs().sum()
            er = 0 if volatility == 0 else change / volatility
            sc = (er * (fastSC - slowSC) + slowSC) ** 2
            kama.iloc[i] = kama.iloc[i - 1] + sc * (close.iloc[i] - kama.iloc[i - 1])
    return kama


# =========================
# CARTEIRA (ATUALIZADA: suporte a adicionar steps e reduzir posição)
# =========================
class Carteira:
    def __init__(self, exchange, paper_trading=False):
        self.posicoes: Dict[str, Any] = {}
        self.historico: List[Dict[str, Any]] = []
        self.lock = Lock()
        self.last_event = None
        self.daily_trades = 0
        self.last_reset_day = datetime.date.today()
        self.exchange = exchange
        self.paper_trading = paper_trading
        self.cash = 0.0
        self.load_or_init()

    def load_or_init(self):
        # Carrega estado: POSITIONS + HISTORY
        self.cash = CONFIG['initial_capital_brl']
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
                if self.paper_trading:
                    self.cash = float(data.get('cash', self.cash))
                self.daily_trades = int(data.get('daily_trades', 0))
                last_reset_str = data.get('last_reset_day')
                if last_reset_str:
                    self.last_reset_day = datetime.datetime.strptime(last_reset_str, "%Y-%m-%d").date()
                logger.info("Estado carregado", posicoes=len(self.posicoes))
            except Exception as e:
                logger.error("Erro ao carregar posições", error=str(e))

        # Carrega Histórico
        # --- FIX: se history.json não existir ou estiver vazio, tenta recuperar do log JSONL (bot_log.jsonl)
        loaded_hist = False
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    self.historico = json.load(f)
                loaded_hist = True
            except Exception:
                self.historico = []
        if not loaded_hist:
            # tenta recuperar do LOG_FILE (jsonl) para reconstruir vendas recentes
            if os.path.exists(LOG_FILE):
                try:
                    recovered = []
                    with open(LOG_FILE, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                j = json.loads(line)
                                msg = str(j.get('message', '')).upper()
                                # heurística: encontrar mensagens de VENDA / SELL / DUST_CLEANED
                                if any(k in msg for k in ('VENDA', 'SELL', 'DUST_CLEANED', 'DUST')):
                                    # tenta recuperar campos extras com pnl/symbol/qty se existirem
                                    symbol = j.get('symbol') or j.get('extra', {}).get('symbol') or None
                                    pnl = j.get('pnl') or j.get('extra', {}).get('pnl') or None
                                    # cria registro parcial simples
                                    rec = {
                                        'symbol': symbol,
                                        'entry_price': None,
                                        'exit_price': None,
                                        'quantity': None,
                                        'pnl': pnl,
                                        'reason': msg,
                                        'timestamp': j.get('timestamp')
                                    }
                                    recovered.append(rec)
                            except Exception:
                                continue
                    if recovered:
                        # adiciona no histórico (no topo)
                        self.historico.extend(recovered)
                        logger.info("Histórico recuperado do log (heurístico)", recovered=len(recovered))
                except Exception:
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
            # --- FIX: garantir escrita consistente do histórico (append-safe)
            try:
                with open(HISTORY_FILE, 'w') as f:
                    json.dump(self.historico, f, indent=2, default=str)
            except Exception as e:
                logger.error("Erro ao salvar history.json", error=str(e))
        except Exception as e:
            logger.error("Erro ao salvar dados", error=str(e))

    def _check_daily_reset(self):
        today = datetime.date.today()
        if today > self.last_reset_day:
            logger.system(f"Novo dia detectado ({today}). Resetando contador de trades.")
            self.daily_trades = 0
            self.last_reset_day = today
            self.salvar()

    def sync_balance(self):
        if self.paper_trading:
            return
        try:
            balance = self.exchange.fetch_balance()
            usdt_free = 0.0
            if isinstance(balance, dict):
                try:
                    usdt_free = float(balance['USDT']['free'])
                except Exception:
                    usdt_free = 0.0
            self.cash = usdt_free
        except Exception as e:
            logger.error("Erro ao sincronizar saldo", error=str(e))

    # Execução da compra de mercado (usada tanto por abrir_posicao quanto por adicionar step)
    def _execute_market_buy(self, symbol, quantity):
        if self.paper_trading:
            return {'filled': quantity, 'average': None}
        order = self.exchange.create_order(symbol, 'market', 'buy', quantity)
        return order

    def abrir_posicao(self, symbol, entry_price, quantity, atr, stop_price=None, take_price=None, entry_plan: Optional[List[Dict]] = None):
        with self.lock:
            self._check_daily_reset()
            cost = float(entry_price) * float(quantity)

            if not self.paper_trading:
                self.sync_balance()

            if cost > self.cash:
                logger.warning("Saldo insuficiente para operação", symbol=symbol, cost=round(cost, 4), cash=round(self.cash, 4))
                return False

            # Executa ordem de mercado real (ou simula em paper)
            if not self.paper_trading:
                try:
                    order = self._execute_market_buy(symbol, quantity)
                    exec_qty = float(order.get('filled') or order.get('amount') or quantity)
                    executed_price = float(order.get('average') or entry_price)
                    # pequeno sleep para sincronizar
                    time.sleep(1)
                    self.sync_balance()
                except Exception as e:
                    logger.error("ERRO CRÍTICO AO COMPRAR NA BINANCE", symbol=symbol, error=str(e))
                    return False
            else:
                exec_qty = float(quantity)
                executed_price = float(entry_price)
                self.cash -= cost

            # cria posição local com metadados para gestão (inclui entry_plan para steps restantes)
            pos = {
                'entry_price': float(executed_price),
                'quantity': float(exec_qty),
                'invested': float(executed_price * exec_qty),
                'atr': float(atr),
                'stop_price': float(stop_price) if stop_price else None,
                'take_price': float(take_price) if take_price else None,
                'trail_stop': None,
                'timestamp': str(datetime.datetime.now()),
                'entry_plan': entry_plan or [],  # lista de pending steps
                'tp_executed': [],  # lista de tiers executadas
                'cooldown_until': None
            }

            self.posicoes[symbol] = pos
            self.last_event = {'symbol': symbol, 'type': 'BUY', 'price': executed_price, 'qty': exec_qty}
            self.daily_trades += 1
            self.salvar()
            logger.success("COMPRA Executada", symbol=symbol, price=round(executed_price, 6), qty=round(exec_qty, 6))
            return True

    def add_to_position(self, symbol, add_price, add_qty, atr):
        """Adiciona (pyramiding) à posição existente: executa market buy e atualiza posição local.
        Retorna True se bem-sucedido; False caso contrário.
        """
        with self.lock:
            if symbol not in self.posicoes:
                return False
            pos = self.posicoes[symbol]
            if not self.paper_trading:
                try:
                    order = self._execute_market_buy(symbol, add_qty)
                    filled = float(order.get('filled') or order.get('amount') or add_qty)
                    avg_price = float(order.get('average') or add_price)
                    time.sleep(1)
                    self.sync_balance()
                except Exception as e:
                    logger.error("Erro ao adicionar posição (API)", symbol=symbol, error=str(e))
                    return False
            else:
                filled = add_qty
                avg_price = add_price
                self.cash -= (avg_price * filled)

            # recalcula entry_price ponderado
            prev_qty = pos.get('quantity', 0.0)
            prev_invested = pos.get('invested', 0.0)
            new_qty = prev_qty + filled
            new_invested = prev_invested + (avg_price * filled)
            pos['quantity'] = float(new_qty)
            pos['invested'] = float(new_invested)
            pos['entry_price'] = float(new_invested / new_qty) if new_qty > 0 else pos['entry_price']
            # atualiza ATR com média simples (pode ser refinado)
            pos['atr'] = float((pos.get('atr', 0.0) + atr) / 2.0)
            pos['timestamp'] = str(datetime.datetime.now())
            self.posicoes[symbol] = pos
            self.salvar()
            logger.info("Step de entrada adicionado", symbol=symbol, added_qty=filled, new_qty=new_qty)
            return True

    # --- FIX: melhorar tratamento de precision/min_amount/NOTIONAL antes de vender ---
    def reduzir_posicao(self, symbol, qty_to_sell, exit_price, reason="TP", current_price=None):
        """Vende uma fração da posição (market sell) e atualiza pos local. Retorna record do fechamento parcial/completo.
        """
        with self.lock:
            self._check_daily_reset()
            if symbol not in self.posicoes:
                return None
            pos = self.posicoes[symbol]
            quantity = float(pos.get('quantity', 0.0))
            if qty_to_sell <= 0 or quantity <= 0:
                return None

            sell_qty = min(qty_to_sell, quantity)

            if not self.paper_trading:
                try:
                    # consulta saldo disponível
                    base_asset = symbol.split('/')[0]
                    bal = {}
                    try:
                        bal = self.exchange.fetch_balance()
                    except Exception:
                        pass
                    available_base = 0.0
                    if isinstance(bal, dict):
                        if base_asset in bal and isinstance(bal[base_asset], dict):
                            available_base = float(bal[base_asset].get('free', 0.0) or 0.0)
                        elif 'free' in bal and isinstance(bal['free'], dict):
                            available_base = float(bal['free'].get(base_asset, 0.0) or 0.0)

                    qty_to_attempt = min(sell_qty, available_base)

                    # small epsilon
                    epsilon = 1e-12
                    qty_to_attempt = max(0.0, qty_to_attempt - epsilon)

                    if qty_to_attempt <= 0:
                        logger.error("Saldo em carteira insuficiente/zerado para vender", symbol=symbol, available=available_base)
                        # força limpeza se for poeira
                        if available_base < 1e-5: 
                             self.posicoes.pop(symbol, None)
                             self.salvar()
                        return None

                    # --- CORREÇÃO DE PRECISÃO E MIN_NOTIONAL ---
                    # Para evitar erro Filter failure: NOTIONAL ou LOT_SIZE
                    market = self.exchange.market(symbol)
                    min_amount, amount_precision, min_cost = self._get_min_lot_size(market)
                    
                    # tenta usar amount_to_precision se disponível
                    try:
                        qty_to_attempt = float(self.exchange.amount_to_precision(symbol, qty_to_attempt))
                    except Exception:
                        # se amount_to_precision falhar, aplicamos fallback manual
                        if amount_precision is not None:
                            step = Decimal('1').scaleb(-amount_precision)
                            dqty = Decimal(str(qty_to_attempt))
                            floored = (dqty // step) * step
                            qty_to_attempt = float(floored.quantize(step, rounding=ROUND_DOWN))

                    # Rechecar min_amount
                    if min_amount is None:
                        # fallback conservador
                        min_amount = 1e-8
                    
                    # --- NOVO: CHECK MIN COST / NOTIONAL ---
                    estimated_value = qty_to_attempt * exit_price
                    # Se tivermos min_cost (ex: 5 USD), verificamos se o valor é menor
                    is_dust = False
                    if min_cost and estimated_value < min_cost:
                        is_dust = True
                    
                    if qty_to_attempt < (min_amount - 1e-12) or is_dust:
                        # Quantidade ou Valor abaixo do permitido pela Binance -> DUST
                        if CONFIG.get('auto_clean_dust', True):
                            # limpamos localmente (dust)
                            logger.warning("Posição considerada POEIRA (Dust/Notional) - Limpando localmente.",
                                           symbol=symbol, qty=qty_to_attempt, val=estimated_value, min_notional=min_cost)
                            record = {
                                'symbol': symbol,
                                'entry_price': pos.get('entry_price'),
                                'exit_price': exit_price,
                                'quantity': float(qty_to_attempt),
                                'pnl': 0.0,
                                'reason': reason + " (DUST_CLEANED)",
                                'timestamp': str(datetime.datetime.now())
                            }
                            # remove posição local
                            self.posicoes.pop(symbol, None)
                            self.historico.append(record)
                            self.last_event = {'symbol': symbol, 'type': 'DUST_CLEANED', 'reason': reason}
                            self.daily_trades += 0
                            self.salvar()
                            return record
                        else:
                            logger.error("Quantidade/Valor < mínimo — pular venda", symbol=symbol, val=estimated_value, min_notional=min_cost)
                            return None

                    # final: cria ordem de venda de mercado com qty_to_attempt
                    order = self.exchange.create_order(symbol, 'market', 'sell', qty_to_attempt)
                    
                    # Se deu certo, atualizamos localmente
                    filled_qty = float(order.get('filled') or order.get('amount') or qty_to_attempt)
                    exec_price = float(order.get('average') or exit_price)
                    time.sleep(1)
                    self.sync_balance()

                except Exception as e:
                    logger.error("ERRO CRÍTICO AO VENDER NA BINANCE (redução)", symbol=symbol, error=str(e))
                    return None
            else:
                # Paper trading
                filled_qty = float(qty_to_sell)
                exec_price = float(exit_price)
                cost = filled_qty * exec_price
                self.cash += cost

            # Atualiza posição local
            remaining_qty = quantity - filled_qty
            entry_price = pos['entry_price']
            
            # PnL do trecho vendido
            pnl = (exec_price - entry_price) * filled_qty
            # Ajusta invested proporcionalmente
            invested_before = pos.get('invested', 0.0)
            invested_after = invested_before * (remaining_qty / quantity) if quantity > 0 else 0.0
            
            record = {
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': exec_price,
                'quantity': filled_qty,
                'pnl': pnl,
                'reason': reason,
                'timestamp': str(datetime.datetime.now())
            }
            self.historico.append(record)
            self.last_event = {'symbol': symbol, 'type': 'SELL', 'price': exec_price, 'pnl': pnl, 'reason': reason}

            if remaining_qty <= (1e-6): # se sobrou quase nada, remove
                self.posicoes.pop(symbol, None)
                logger.success("Posição FECHADA", symbol=symbol, pnl=round(pnl, 4), reason=reason)
            else:
                pos['quantity'] = float(remaining_qty)
                pos['invested'] = float(invested_after)
                self.posicoes[symbol] = pos
                logger.success("Posição REDUZIDA", symbol=symbol, sold=filled_qty, left=remaining_qty, pnl=round(pnl, 4))

            self.salvar()
            return record

    def _get_min_lot_size(self, market):
        # Tenta extrair min amount, precision e min notional (cost)
        try:
            min_amount = None
            amount_precision = None
            min_cost = None # Notional

            if 'limits' in market:
                limits = market['limits']
                min_amount = limits.get('amount', {}).get('min')
                min_cost = limits.get('cost', {}).get('min') # Notional limit (e.g. 5$)
                if min_cost is None:
                     # fallback location
                     min_cost = limits.get('market', {}).get('min')

            if 'precision' in market:
                amount_precision = market['precision'].get('amount')

            # conversions
            if min_amount is not None: min_amount = float(min_amount)
            if min_cost is not None: min_cost = float(min_cost)
            
            if amount_precision is not None:
                try:
                    amount_precision = int(amount_precision)
                except:
                    amount_precision = int(float(amount_precision))
            
            return min_amount, amount_precision, min_cost
        except Exception:
            pass
        return None, None, None

    def _save_sample(self, symbol: str, features: Dict[str, Any], label: int):
        try:
            path = os.path.join(SAMPLES_DIR, f"{symbol.replace('/', '_')}_samples.jsonl")
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'ts': str(datetime.datetime.now()), 'features': features, 'label': int(label)}) + "\n")
        except Exception:
            pass


# =========================
# METRICS
# =========================
class Metrics:
    def __init__(self):
        self.path = METRICS_FILE
        self.lock = Lock()

    def record_trade(self, pnl):
        with self.lock:
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}

            today = datetime.date.today().isoformat()
            if today not in data:
                data[today] = {'trades': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0}

            data[today]['trades'] += 1
            data[today]['pnl'] += pnl
            if pnl > 0:
                data[today]['wins'] += 1
            else:
                data[today]['losses'] += 1

            try:
                with open(self.path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass


metrics = Metrics()

# =========================
# ROBOTRADER (Cérebro) - INCLUI OPÇÃO C
# =========================
class RoboTrader:
    def __init__(self):
        self.running = False
        self.live_state = self._load_live_state()
        self.models = {}  # sym -> {'model': model, 'features': [...], 'partial_fit': bool}
        self.market_prices = {}
        self.cooldowns = {}
        self.equity_history = []
        self.equity_high = 0.0
        self.equity_low = float('inf')

        # --- SETUP BINANCE ---
        if not CONFIG['paper_trading']:
            api_key = os.getenv('BINANCE_API_KEY')
            secret = os.getenv('BINANCE_SECRET_KEY')

            if not api_key or not secret:
                logger.error("❌ ERRO: Chaves API não encontradas no arquivo .env")
                logger.info("Crie um arquivo .env com BINANCE_API_KEY e BINANCE_SECRET_KEY")
                sys.exit(1)

            try:
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                balance = self.exchange.fetch_balance()
                free_usdt = 0.0
                if isinstance(balance, dict):
                    try:
                        free_usdt = float(balance['USDT']['free'])
                    except Exception:
                        free_usdt = 0.0
                logger.system(f"✅ CONECTADO À BINANCE! Saldo Livre: ${free_usdt} USDT")
            except Exception as e:
                logger.error("Falha ao conectar na Binance", error=str(e))
                sys.exit(1)
        else:
            self.exchange = ccxt.binance({'enableRateLimit': True})
            logger.system("Modo Paper Trading (Simulação)")

        try:
            self.exchange.load_markets(True)
        except Exception:
            logger.warning("Falha ao carregar markets da exchange (continuando com fallback)")

        self.carteira = Carteira(self.exchange, paper_trading=CONFIG['paper_trading'])

        if not CONFIG['paper_trading']:
            self.carteira.sync_balance()

        if self.live_state.get("armed"):
            pass

        self.load_models()
        logger.system("Sistema Inicializado", mode="REAL" if not CONFIG['paper_trading'] else "PAPER")

    def _load_live_state(self):
        if os.path.exists(LIVE_STATE_FILE):
            try:
                with open(LIVE_STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"armed": False}

    def load_models(self):
        count = 0
        for sym in CONFIG['symbols']:
            safe_sym = sym.replace('/', '_')
            model_path = os.path.join(INTEL_DIR, f"{safe_sym}.joblib")
            meta_path = os.path.join(META_DIR, f"{safe_sym}_meta.json")
            if os.path.exists(model_path) and os.path.exists(meta_path):
                try:
                    loaded = joblib.load(model_path)
                    meta = json.load(open(meta_path))
                    supports_partial = False
                    if SKLEARN_AVAILABLE and hasattr(loaded, 'partial_fit'):
                        supports_partial = True
                    self.models[sym] = {
                        'model': loaded,
                        'features': meta.get('features', []),
                        'partial_fit': supports_partial
                    }
                    count += 1
                except Exception:
                    pass
        logger.info(f"{count} Modelos de IA carregados.")

    @retry_on_exception(max_attempts=3)
    def safe_fetch_data(self, symbol, limit=200):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=CONFIG['timeframe'], limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        return df

    @retry_on_exception(max_attempts=3)
    def safe_ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    # FUNÇÃO ANALYZE COMPLETA (Gera todas as features para a IA) - agora inclui KAMA
    def analyze(self, symbol):
        try:
            df = self.safe_fetch_data(symbol, limit=200)
            if df is None or df.empty:
                return None

            df['c'] = df['c'].astype(float)
            df['h'] = df['h'].astype(float)
            df['l'] = df['l'].astype(float)
            df['v'] = df['v'].astype(float)
            current_close = df['c'].iloc[-1]
            self.market_prices[symbol] = current_close

            # 1. INDICADORES BASE
            df['rsi'] = ta.rsi(df['c'], length=14)
            df['ema50'] = ta.ema(df['c'], length=50)
            df['atr'] = ta.atr(df['h'], df['l'], df['c'], length=14)

            # 1.b KAMA (Média Adaptativa)
            ama_params = CONFIG.get('adaptive_ma', {'n': 10, 'fast': 2, 'slow': 30})
            try:
                kama_ser = kaufman_adaptive_moving_average(df['c'], n=ama_params.get('n', 10), fast=ama_params.get('fast', 2), slow=ama_params.get('slow', 30))
                df['kama'] = kama_ser
            except Exception:
                df['kama'] = df['c'].rolling(10, min_periods=1).mean()

            # 2. FEATURES PARA O MODELO (Conforme seus meta.json)
            df['rsi_norm'] = df['rsi'] / 100.0
            df['dist_ema'] = (df['c'] - df['ema50']) / df['ema50']
            macd = ta.macd(df['c'])
            if isinstance(macd, pd.DataFrame) and 'MACD_12_26_9' in macd.columns:
                df['macd'] = macd['MACD_12_26_9']
            else:
                df['macd'] = 0.0
            bb = ta.bbands(df['c'], length=20, std=2)
            bbu_col = [c for c in bb.columns if 'BBU' in c]
            df['bb_upper'] = bb[bbu_col[0]] if bbu_col else df['c']
            df['dist_bb_up'] = (df['bb_upper'] - df['c']) / df['c']
            df['vol_change'] = df['v'].pct_change().fillna(0.0)

            row = df.iloc[-1].to_dict()
            return row
        except Exception as e:
            logger.error("Erro no analyze", symbol=symbol, error=str(e))
            return None

    def _compute_drawdown(self, equity_value: float):
        if equity_value > self.equity_high:
            self.equity_high = equity_value
        if self.equity_high <= 0:
            return 0.0
        drawdown = max(0.0, (self.equity_high - equity_value) / max(1e-8, self.equity_high))
        return drawdown

    def _expected_value_per_unit(self, p_win: float, atr: float):
        win_amount = CONFIG['take_profit_mult'] * atr
        loss_amount = CONFIG['stop_loss_mult'] * atr
        ev = p_win * win_amount - (1 - p_win) * loss_amount
        return ev, win_amount, loss_amount

    # ---- helper: cria plano de entrada escalonada (entry ladder)
    def _plan_entry_ladder(self, price: float, atr: float, total_allocation_usd: float) -> List[Dict[str, Any]]:
        steps = CONFIG.get('entry_steps', [0.5, 0.3, 0.2])
        spacing = CONFIG.get('entry_step_spacing_atr', [0.0, 1.5, 2.5])
        # garante o mesmo tamanho
        if len(spacing) < len(steps):
            # estende com últimos valores
            spacing = spacing + [spacing[-1]] * (len(steps) - len(spacing))

        plan = []
        for frac, sp in zip(steps, spacing):
            step_alloc = total_allocation_usd * frac
            step_price = max(0.0, price - (sp * atr))
            qty = step_alloc / step_price if step_price > 0 else 0.0
            plan.append({'price_level': step_price, 'portion': frac, 'alloc_usd': step_alloc, 'qty': qty, 'done': False})
        return plan

    #
    # ---------------------- NEW HELPER: SAFE MODEL PREDICTION ----------------------
    #
    # Esse helper garante que o modelo receba um DataFrame com as mesmas colunas (e na mesma ordem)
    # com que foi treinado (se disponível via model.feature_names_in_). Caso contrário, usa
    # a lista de features carregada do meta.json. Preenche colunas ausentes com 0.
    #
    def _predict_proba_safe(self, model, meta_features: List[str], row: Dict[str, Any]):
        """
        Retorna array-like forecast de predict_proba (2d). Em caso de falha, retorna [[0.5,0.5]] como fallback.
        """
        try:
            # Prioriza feature_names_in_ se disponível (sklearn)
            if hasattr(model, 'feature_names_in_') and getattr(model, 'feature_names_in_', None) is not None:
                feature_names = list(model.feature_names_in_)
            else:
                feature_names = list(meta_features or [])

            if not feature_names:
                # se não temos nomes, tentamos construir a data frame com as keys do row (fallback)
                X = pd.DataFrame([row])
            else:
                # garante que todas as colunas existam e na ordem correta
                data = {fn: row.get(fn, 0) for fn in feature_names}
                X = pd.DataFrame([data], columns=feature_names)

            # força conversão numérica quando possível
            for c in X.columns:
                try:
                    X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)
                except Exception:
                    pass

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                return proba
            else:
                # modelo não tem predict_proba: tenta predict e converte para probabilidade artificial
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    # transforma 0/1 em prob 0.0/1.0 (fallback)
                    p = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
                    return [[1.0 - p, p]]
        except Exception:
            # não crashar modelo; retornar fallback neutro
            return [[0.5, 0.5]]

        return [[0.5, 0.5]]

    #
    # -------------------------------------------------------------------------------
    #

    def cycle(self):
        try:
            logger.info(f"--- Ciclo iniciado (Cash: {round(self.carteira.cash,4)} USDT) ---")
        except Exception:
            logger.info("--- Ciclo iniciado ---")

        # Atualiza equity
        invested_val = 0.0
        for sym, pos in self.carteira.posicoes.items():
            curr_price = self.market_prices.get(sym, pos['entry_price'])
            invested_val += (pos['quantity'] * curr_price)

        total_equity = self.carteira.cash + invested_val
        drawdown = self._compute_drawdown(total_equity)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        ts_now = now_utc.strftime('%H:%M:%S')
        self.equity_history.append({'time': ts_now, 'value': total_equity, 'drawdown': drawdown})
        if len(self.equity_history) > 200:
            self.equity_history.pop(0)

        if drawdown >= CONFIG.get('max_drawdown_pct', 0.99):
            logger.warning("Pausando novas entradas: drawdown crítico atingido", drawdown=drawdown)
            allow_entries = False
        else:
            allow_entries = True

        # Gerenciar posições: trailing stops, break-even, partial TPs, execução de steps
        for sym in list(self.carteira.posicoes.keys()):
            try:
                tk = self.safe_ticker(sym)
                price = tk.get('last', None) if isinstance(tk, dict) else None
                if price is None:
                    continue
                self.market_prices[sym] = price
                pos = self.carteira.posicoes[sym]

                # --- LÓGICA HOLD-FOREVER (NÃO VENDER NO PREJUÍZO) ---
                # Removemos a lógica antiga de Stop Loss / Trailing Stop automático que gerava perda
                # O robô agora só vende se houver LUCRO (Lock Profit) ou se atingir Take Profit positivo
                
                # LOCK PROFIT (Venda com Lucro)
                entry_price = pos.get('entry_price', 0.0)
                quantity = pos.get('quantity', 0.0)
                invested = pos.get('invested', entry_price * quantity)
                unreal_pnl = (price - entry_price) * quantity
                unreal_pct = (unreal_pnl / invested) if invested and invested > 0 else 0.0

                min_profit_usd = CONFIG.get('min_profit_usd', 0.50)
                min_profit_pct = CONFIG.get('min_profit_pct', 0.015)

                reason = None
                
                # Só vende se estiver no lucro e acima dos mínimos configurados
                if unreal_pnl >= min_profit_usd or unreal_pct >= min_profit_pct:
                    reason = "LOCK PROFIT (LUCRO ATINGIDO)"

                # Fallback: Take Profit Tradicional (Preço Alvo) - Também é lucro
                take_price = pos.get('take_price')
                if take_price and price >= take_price:
                     # Dupla verificação: só executa TP se o PnL for positivo
                     if unreal_pnl > 0:
                        reason = "TAKE PROFIT (ALVO ATINGIDO)"

                if reason:
                    # Se caiu aqui, é lucro garantido
                    rec = self.carteira.reduzir_posicao(sym, pos.get('quantity', 0.0), price, reason=reason)
                    if rec:
                        metrics.record_trade(rec['pnl'])
                
                # NOTA: Removemos os blocos "else if stop_loss" e "else if trailing_stop"
                # Assim, se a moeda cair, o robô não faz NADA. Ele segura (Hold) até subir.

            except Exception:
                # não poluir logs
                pass

        # Novas Entradas
        self.carteira._check_daily_reset()
        # ---------- DAILY LIMIT DISABLED ----------
        # A verificação de limite diário original foi removida para permitir operação 24/7.
        # Mantemos o contador self.carteira.daily_trades para relatório/monitoramento, mas
        # o robô NÃO irá interromper entradas por atingir um limite diário.
        #
        # (trecho original comentado)
        # if self.carteira.daily_trades >= CONFIG['max_daily_trades']:
        #     logger.info("Limite diário de trades atingido", daily_trades=self.carteira.daily_trades)
        #     return
        # ------------------------------------------

        if not allow_entries:
            return

        slots = max(0, CONFIG['max_positions'] - len(self.carteira.posicoes))
        if slots <= 0:
            return

        # Roda análise e pontuação
        all_candidates: List[Dict[str, Any]] = []
        for sym in CONFIG['symbols']:
            try:
                if sym in self.carteira.posicoes:
                    # se existe posição com entry_plan pendente, entregamos execução mais acima
                    continue
                if self.cooldowns.get(sym) and datetime.datetime.now() < self.cooldowns[sym]:
                    continue

                row = self.analyze(sym)
                if not row:
                    continue

                price = float(row.get('c') or 0.0)
                atr = float(row.get('atr') or 0.0)
                if atr <= 0:
                    atr = max(price * 0.005, 1e-8)

                # IA / model scoring
                score = 0.0
                used_model = 'NONE'
                p_win = 0.5
                if sym in self.models:
                    try:
                        model_data = self.models[sym]
                        features = model_data.get('features', [])
                        model = model_data['model']

                        # <<< USAR O HELPER SEGURO PARA PREDIÇÃO (evita warnings do sklearn) >>>
                        if hasattr(model, 'predict_proba'):
                            proba = self._predict_proba_safe(model, features, row)
                            try:
                                p_win = float(proba[0][1])
                            except Exception:
                                p_win = 0.5
                            score = p_win
                        elif hasattr(model, 'predict'):
                            # fallback: monta DataFrame com as features (ordem do meta)
                            X = pd.DataFrame([{k: row.get(k, 0) for k in features}])
                            score = float(model.predict(X)[0])
                            p_win = score
                        else:
                            p_win = 0.5
                            score = 0.5
                        used_model = 'USADO'
                    except Exception as e:
                        p_win = 0.5
                        score = 0.5
                        used_model = 'FALLBACK'
                        if CONFIG['verbose_logs']:
                            logger.error("Falha na previsão da IA. Usando fallback.", symbol=sym, error=str(e))
                else:
                    score = 0.5
                    p_win = 0.5
                    if row.get('rsi', 50.0) < 35:
                        score += 0.15
                        p_win += 0.08
                    if row.get('ema50') and row['c'] > row['ema50']:
                        score += 0.08
                        p_win += 0.03
                    # filtro adicional: preço acima KAMA preferível
                    if 'kama' in row and row['c'] > row['kama']:
                        score += 0.05
                        p_win += 0.02
                    used_model = 'FALLBACK'

                ev_per_unit, win_amount, loss_amount = self._expected_value_per_unit(p_win, atr)
                ev_per_usd = ev_per_unit / max(price, 1e-8)

                if CONFIG['verbose_logs']:
                    logger.info("Score IA", symbol=sym, score=f"{score:.4f}", model=used_model, p_win=round(p_win,4), ev_per_unit=round(ev_per_unit,6))

                all_candidates.append({
                    'symbol': sym,
                    'score': float(score),
                    'p_win': float(p_win),
                    'price': price,
                    'atr': atr,
                    'ev_per_unit': float(ev_per_unit),
                    'ev_per_usd': float(ev_per_usd),
                    'row': row
                })
            except Exception:
                continue

        if not all_candidates:
            if CONFIG['verbose_logs']:
                logger.info("Nenhum candidato analisado neste ciclo.")
            return

        all_candidates.sort(key=lambda x: x['score'], reverse=True)

        thresholds = [CONFIG.get('min_confidence', 0.55), 0.50, 0.45, 0.40]
        selected_candidates: List[Dict[str, Any]] = []
        used_threshold = None
        for thr in thresholds:
            selected_candidates = [c for c in all_candidates if c['score'] > thr and c['ev_per_usd'] >= CONFIG.get('min_expected_ev', 0.0)]
            if selected_candidates:
                used_threshold = thr
                logger.info("Candidatos encontrados (com EV positivo)", threshold=thr, found=len(selected_candidates))
                break

        if not selected_candidates:
            top = all_candidates[0]
            if top['score'] >= 0.40 and top['ev_per_usd'] >= CONFIG.get('min_expected_ev', 0.0):
                selected_candidates = [top]
                used_threshold = 'forced_top_>=0.40_with_EV_check'
                logger.warning("Forçando melhor candidato (>=0.40) com EV positivo", symbol=top['symbol'], score=top['score'], ev_per_usd=top['ev_per_usd'])
            else:
                logger.info("Nenhum candidato qualificado encontrado neste ciclo", best_score=all_candidates[0]['score'], best_ev=all_candidates[0]['ev_per_usd'])
                return

        selected_candidates = selected_candidates[:slots]

        # ALOCAÇÃO (mantém buffer e respeita exposição máxima)
        available_cash = max(0.0, self.carteira.cash * (1.0 - CONFIG['allocation_buffer_pct']))
        current_exposure = sum([p['invested'] for p in self.carteira.posicoes.values()]) if self.carteira.posicoes else 0.0
        max_allowed_exposure = total_equity * CONFIG['max_exposure_pct']
        remaining_exposure_capacity = max_allowed_exposure - current_exposure
        if remaining_exposure_capacity <= 0:
            logger.warning("Capacidade de exposição esgotada", current_exposure=current_exposure, max_allowed=max_allowed_exposure)
            return

        available_cash = min(available_cash, remaining_exposure_capacity, self.carteira.cash)

        if available_cash < CONFIG['min_trade_usd']:
            logger.info("Cash insuficiente para novas alocações", available_cash=round(available_cash, 4))
            return

        total_score = sum(max(0.0, c['ev_per_usd']) for c in selected_candidates) or sum(c['score'] for c in selected_candidates) or 1.0
        allocations = []
        
        # --- NOVA LÓGICA DE DIVERSIFICAÇÃO ---
        # Define o teto máximo em Dólares baseado na porcentagem configurada
        max_alloc_usd = total_equity * CONFIG.get('max_position_pct', 1.0)

        for c in selected_candidates:
             # 1. Cálculo inicial (divisão do cash disponível ou proporcional)
            if CONFIG['allocation_mode'] == 'equal':
                alloc = available_cash / len(selected_candidates)
            else:
                weight = max(0.0, c['ev_per_usd'])
                if weight == 0:
                    weight = c['score']
                alloc = available_cash * (weight / total_score)
            
            # 2. APLICA O TETO (Força diversificação)
            if alloc > max_alloc_usd:
                alloc = max_alloc_usd

            # 3. VERIFICA MÍNIMO DA BINANCE (Proteção contra ordens inválidas)
            # Se a alocação ficou abaixo de ~6 USD, tentamos arredondar para o mínimo se houver saldo
            if alloc < CONFIG['min_allocation_per_slot']:
                # Se temos saldo suficiente para cobrir o mínimo, usamos o mínimo
                if available_cash >= CONFIG['min_allocation_per_slot']:
                    alloc = CONFIG['min_allocation_per_slot']
                else:
                    # Se não tem nem o mínimo, não entra
                    alloc = 0.0
            
            # 4. Verifica se ainda sobrou cash real para essa alocação (double check)
            if alloc > available_cash:
                alloc = available_cash

            if alloc > 0:
                allocations.append((c, alloc))
                # Deduz do disponível virtualmente para o loop continuar certo
                available_cash -= alloc 

        # Executa entradas
        for cand, alloc_usd in allocations:
            if alloc_usd <= 0:
                continue
            
            sym = cand['symbol']
            price = cand['price']
            atr = cand['atr']
            
            # define stop e take
            # NOTA: O stop_dist ainda é calculado para registro interno, 
            # mas como removemos a lógica de disparo no 'cycle', ele é ignorado na prática.
            stop_dist = CONFIG['stop_loss_mult'] * atr
            take_dist = CONFIG['take_profit_mult'] * atr
            stop_price = price - stop_dist
            take_price = price + take_dist

            # entry steps logic
            # se tivermos entry_steps > 1, alocamos apenas a primeira parte
            plan = self._plan_entry_ladder(price, atr, alloc_usd)
            first_step = plan[0]
            
            # Executa primeira compra
            if self.carteira.abrir_posicao(sym, price, first_step['qty'], atr, stop_price, take_price, entry_plan=plan[1:]):
                # Grupo cooldown
                grp = SYMBOL_TO_GROUP.get(sym, 'ungrouped')
                if grp != 'ungrouped':
                    for s_in_grp in CONFIG['groups'].get(grp, []):
                        self.cooldowns[s_in_grp] = datetime.datetime.now() + datetime.timedelta(seconds=CONFIG['group_cooldown_seconds'])
                
                # Online Learning sample
                if CONFIG['online_learning'] and sym in self.models and self.models[sym]['partial_fit']:
                    try:
                        # Exemplo de label 1 (comprou). O label real virá no futuro (se deu lucro ou nao)
                        # Aqui apenas salvamos o feature vector para uso posterior se quisermos
                        pass
                    except Exception:
                        pass

    def run(self):
        self.running = True
        logger.system("🚀 Robô iniciado! Aguardando ciclos...")
        while self.running:
            if self.live_state.get("armed"):
                try:
                    self.cycle()
                except Exception as e:
                    logger.error("Erro no ciclo principal", error=str(e))
            else:
                logger.info("Robô desarmado. Aguardando comando de START...")
            
            time.sleep(CONFIG['sleep_cycle'])

    def stop(self):
        self.running = False
        logger.system("🛑 Parando robô...")


# =========================
# FLASK SERVER
# =========================
app = Flask(__name__)
CORS(app)

robo = RoboTrader()

@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        # Prepara posições para JSON
        pos_list = []
        total_pnl_open = 0.0
        
        # Atualiza preços atuais para PnL em tempo real
        for sym, p in robo.carteira.posicoes.items():
            curr_price = robo.market_prices.get(sym, p['entry_price'])
            qty = p['quantity']
            invested = p.get('invested', p['entry_price'] * qty)
            val_now = qty * curr_price
            pnl = val_now - invested
            total_pnl_open += pnl
            
            pos_list.append({
                'symbol': sym,
                'entryPrice': p['entry_price'],
                'quantity': qty,
                'invested': invested,
                'currentPrice': curr_price,
                'pnl': pnl,
                'stopPrice': p.get('stop_price'),
                'takePrice': p.get('take_price')
            })

        # calcula equity real time
        balance = robo.carteira.cash
        equity = balance + sum([p['invested'] + p['pnl'] for p in pos_list])

        return jsonify({
            'isRunning': robo.live_state.get("armed", False),
            'balance': balance,
            'equity': equity,
            'openPositions': pos_list,
            'dailyTrades': robo.carteira.daily_trades,
            'totalTrades': len(robo.carteira.historico),
            'marketPrices': robo.market_prices,
            'lastEvent': robo.carteira.last_event
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_bot():
    robo.live_state["armed"] = True
    with open(LIVE_STATE_FILE, 'w') as f:
        json.dump(robo.live_state, f)
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    robo.live_state["armed"] = False
    with open(LIVE_STATE_FILE, 'w') as f:
        json.dump(robo.live_state, f)
    return jsonify({'status': 'stopped'})

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(robo.carteira.historico)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    return jsonify(logger.buffer)

# --- ENDPOINTS MANUAIS ---

@app.route('/api/close_position', methods=['POST'])
def api_close_position():
    """
    Endpoint para fechar uma posição manualmente (Venda a Mercado).
    """
    try:
        data = request.json
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({'status': 'error', 'message': 'Símbolo não informado'}), 400

        # Verifica se a posição existe na carteira do robô
        if symbol not in robo.carteira.posicoes:
            return jsonify({'status': 'error', 'message': 'Posição não encontrada'}), 404

        pos = robo.carteira.posicoes[symbol]
        quantity = pos.get('quantity', 0.0)

        logger.info(f"Recebida solicitação manual de venda para {symbol}", quantity=quantity)

        # Chama a função interna de reduzir_posicao (vende 100%)
        # A flag "MANUAL_USER" ajuda a identificar no histórico
        result = robo.carteira.reduzir_posicao(
            symbol=symbol, 
            qty_to_sell=quantity, 
            exit_price=robo.market_prices.get(symbol, pos['entry_price']), # Tenta preço atual ou usa entrada como ref
            reason="MANUAL_USER"
        )

        if result:
             return jsonify({
                'status': 'success', 
                'message': f'Ordem de venda enviada para {symbol}',
                'details': result
            })
        else:
            # Se retornou None/False, pode ter sido erro de Dust ou API
            return jsonify({'status': 'error', 'message': 'Falha ao executar venda (verifique logs/saldo)'}), 500

    except Exception as e:
        logger.error("Erro no endpoint manual close", error=str(e))
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- NOVO ENDPOINT DE COMPRA MANUAL ---
@app.route('/api/open_position', methods=['POST'])
def api_open_position():
    """
    Endpoint para abrir uma posição manualmente (Compra a Mercado).
    """
    try:
        data = request.json
        symbol = data.get('symbol', '').upper()
        # Usa valor enviado ou o mínimo da config (Default)
        try:
             amount_usd = float(data.get('amount', CONFIG['min_trade_usd']))
        except:
             amount_usd = CONFIG['min_trade_usd']

        if not symbol:
            return jsonify({'status': 'error', 'message': 'Símbolo inválido'}), 400
        
        # Garante formato correto (ex: BTC -> BTC/USDT)
        if not '/' in symbol:
            symbol = f"{symbol}/USDT"

        # Verifica se já está posicionado
        if symbol in robo.carteira.posicoes:
            return jsonify({'status': 'error', 'message': f'Já existe uma posição aberta para {symbol}'}), 400

        logger.info(f"Recebida solicitação manual de COMPRA para {symbol} (${amount_usd})")

        # 1. Analisa o ativo para pegar Preço e ATR atuais
        row = robo.analyze(symbol)
        if not row:
            return jsonify({'status': 'error', 'message': 'Não foi possível obter dados de mercado (Analyze falhou)'}), 500
        
        price = float(row.get('c'))
        atr = float(row.get('atr') or (price * 0.01)) # Fallback ATR

        # 2. Calcula quantidade baseada no valor em Dólar ($)
        if price <= 0:
             return jsonify({'status': 'error', 'message': 'Preço inválido (zero)'}), 500
             
        quantity = amount_usd / price

        # 3. Define Stops (Baseado na config global)
        stop_dist = CONFIG['stop_loss_mult'] * atr
        take_dist = CONFIG['take_profit_mult'] * atr
        stop_price = price - stop_dist
        take_price = price + take_dist

        # 4. Executa a ordem
        success = robo.carteira.abrir_posicao(
            symbol=symbol,
            entry_price=price,
            quantity=quantity,
            atr=atr,
            stop_price=stop_price,
            take_price=take_price
        )

        if success:
            # Opcional: Adiciona cooldown para evitar compra dupla automática imediata
            robo.cooldowns[symbol] = datetime.datetime.now() + datetime.timedelta(seconds=60)
            return jsonify({
                'status': 'success', 
                'message': f'Ordem de COMPRA enviada para {symbol}',
                'details': {'price': price, 'qty': quantity}
            })
        else:
            return jsonify({'status': 'error', 'message': 'Falha ao executar compra (Verifique saldo ou logs)'}), 500

    except Exception as e:
        logger.error("Erro no endpoint manual open", error=str(e))
        return jsonify({'status': 'error', 'message': str(e)}), 500

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Thread do Robô
    t_robo = Thread(target=robo.run)
    t_robo.daemon = True
    t_robo.start()

    # Thread do Flask
    # (ou rodar flask na main thread)
    print("🔥 Servidor rodando na porta 5000...")
    run_flask()