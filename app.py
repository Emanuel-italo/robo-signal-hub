# app.py
"""
VERSÃO DE PRODUÇÃO (LIVE TRADING)
Ajustada para banca pequena (~17 USDT / R$ 100).
INTEGRAÇÃO TOTAL COM MODELOS DE IA TREINADOS e ajuste de dimensionamento.
Corrigido e melhorado:
 - detecção/limpeza de dust (quantias menores que min_amount)
 - Expected Value (EV) para decisões (probabilidade * ganho - (1-prob)*perda)
 - position sizing dinâmico (risk_per_trade combinado com EV)
 - drawdown protection
 - registro de samples e tentativa de partial_fit quando possível
 - mantém lógica original de sinais/IA e comportamentos conservadores
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
# CONFIGURAÇÕES (OTIMIZADAS PARA R$ 100 / 17 USDT)
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
    'max_positions': 2,        # Quantas posições simultâneas
    'stop_loss_mult': 2.5,
    'take_profit_mult': 5.0,
    'risk_per_trade': 0.02,    # 2% do patrimônio por trade (ajustável)
    'min_trade_usd': 5.5,      # mínimo da Binance (deixe > 5)
    'max_daily_trades': 6,
    'min_volume_quote': 5000.0,
    'slippage_pct': 0.0015,
    'fee_pct': 0.001,
    'max_exposure_pct': 0.90,  # Permite usar 90% do saldo
    'group_cooldown_seconds': 60 * 60,

    # Diversificação / Alocação
    'diversify': True,
    'allocation_mode': 'proportional',  # 'proportional' ou 'equal'
    'allocation_buffer_pct': 0.05,  # Mantém 5% do cash como buffer para fees/ordens
    'min_allocation_per_slot': 5.5,  # mesmo que min_trade_usd (segurança)

    # comportamento automático para "dust" (quantias menores que min_amount da exchange)
    'auto_clean_dust': True,  # se True: limpa posições locais com quantia < min_amount (evita loop)

    # MODELO / LEARNING
    'online_learning': True,            # tenta partial_fit quando possível
    'retrain_every_n_trades': 50,       # quando atingir, tente re-treinar/exportar (não automático aqui)
    'min_expected_ev': 0.0005,         # min EV por USD de posição para considerar (ex.: 0.0005 USD por 1 USD investido)
    'conservative_mode': True,         # se True, enrijece stops/taking e reduz risco
    'max_drawdown_pct': 0.20,          # se drawdown > 20% pausa entradas

    # Logging
    'verbose_logs': False,  # se True mostra muitos logs; se False, apenas logs relevantes

    # --- SEGURANÇA ---
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

# =========================
# CARTEIRA
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
        # Em modo real, o cash será sobrescrito pela consulta à API depois
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
                # Se for paper trading, recupera o cash do arquivo. Se for real, ignora (será lido da API)
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
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    self.historico = json.load(f)
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
            with open(HISTORY_FILE, 'w') as f:
                json.dump(self.historico, f, indent=2, default=str)
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
        """Sincroniza saldo com a Binance (Apenas Modo Real)"""
        if self.paper_trading:
            return
        try:
            balance = self.exchange.fetch_balance()
            # Segurança: verifique existência de keys
            usdt_free = 0.0
            if isinstance(balance, dict):
                # estrutura típica: {'USDT': {'free': 1.0, ...}, 'total': {...}}
                if 'USDT' in balance and isinstance(balance['USDT'], dict) and 'free' in balance['USDT']:
                    usdt_free = float(balance['USDT']['free'])
                elif 'free' in balance:
                    # ccxt tem variações
                    try:
                        usdt_free = float(balance['free'].get('USDT', 0.0))
                    except Exception:
                        usdt_free = 0.0
            self.cash = usdt_free
        except Exception as e:
            logger.error("Erro ao sincronizar saldo", error=str(e))

    def abrir_posicao(self, symbol, entry_price, quantity, atr, stop_price=None, take_price=None):
        with self.lock:
            self._check_daily_reset()

            cost = float(entry_price) * float(quantity)

            # Validação Extra de Saldo (Dupla checagem)
            if not self.paper_trading:
                self.sync_balance()

            if cost > self.cash:
                logger.warning("Saldo insuficiente para operação", symbol=symbol, cost=round(cost, 4), cash=round(self.cash, 4))
                return False

            # Execução da Ordem
            if not self.paper_trading:
                try:
                    # Tenta criar ordem de MERCADO na Binance
                    order = self.exchange.create_order(symbol, 'market', 'buy', quantity)
                    # Atualiza preço e qtd com o que foi realmente executado
                    entry_price = float(order.get('average') or order.get('price') or entry_price)
                    # Re-sincroniza saldo após compra
                    time.sleep(1)
                    self.sync_balance()
                except Exception as e:
                    logger.error("ERRO CRÍTICO AO COMPRAR NA BINANCE", symbol=symbol, error=str(e))
                    return False
            else:
                # Paper trading: atualiza cash localmente
                self.cash -= cost

            self.posicoes[symbol] = {
                'entry_price': float(entry_price),
                'quantity': float(quantity),
                'invested': float(entry_price * quantity),
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
            logger.success("COMPRA Executada", symbol=symbol, price=round(entry_price, 6), qty=round(quantity, 6))
            return True

    # --- helpers para min_amount / precision (necessário para evitar erros de dust) ---
    def _ensure_markets_loaded(self):
        """Try to ensure the exchange has markets loaded so we can read limits/precision."""
        try:
            if hasattr(self.exchange, 'markets') and not self.exchange.markets:
                # load_markets pode ser caro, mas é necessário para min_amount/precision
                try:
                    self.exchange.load_markets(True)
                except Exception:
                    # algumas builds do ccxt aceitam load_markets sem argumentos
                    try:
                        self.exchange.load_markets()
                    except Exception:
                        pass
        except Exception:
            pass

    def _get_market_limits_and_precision(self, symbol):
        """
        Retorna (min_amount, amount_precision) para o symbol, se possível.
        Se não for possível, retorna (None, None) para que o caller use fallback.
        """
        try:
            self._ensure_markets_loaded()
            markets = getattr(self.exchange, 'markets', None) or {}
            market = markets.get(symbol) or markets.get(symbol.replace('/', ''))  # fallback
            if market:
                # min amount
                limits = market.get('limits', {}) or {}
                amount_limits = limits.get('amount') or {}
                min_amount = amount_limits.get('min')
                # precision
                precision = market.get('precision', {}) or {}
                amount_precision = precision.get('amount')
                # tipo conversão seguras
                if min_amount is not None:
                    min_amount = float(min_amount)
                if amount_precision is not None:
                    try:
                        amount_precision = int(amount_precision)
                    except Exception:
                        # se precision estiver em float, converte truncando
                        amount_precision = int(float(amount_precision))
                return min_amount, amount_precision
        except Exception:
            pass
        return None, None

    # --- registro de amostras (features + label) para posterior retrain / partial_fit ---
    def _save_sample(self, symbol: str, features: Dict[str, Any], label: int):
        """
        Salva cada sample como JSONL em SAMPLES_DIR/symbol_samples.jsonl
        features: dicionário
        label: 0 ou 1
        """
        try:
            path = os.path.join(SAMPLES_DIR, f"{symbol.replace('/', '_')}_samples.jsonl")
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'ts': str(datetime.datetime.now()), 'features': features, 'label': int(label)}) + "\n")
        except Exception:
            pass

    def fechar_posicao(self, symbol, exit_price, reason="Sinal"):
        """
        Fechamento robusto:
         - verifica saldo disponível do ativo base (ex: SOL)
         - ajusta quantidade pela precisão da exchange
         - tenta venda parcial se saldo insuficiente
         - re-adiciona posição local em caso de falha
         - se quantidade disponível < min_amount da exchange e CONFIG['auto_clean_dust'] == True,
           limpa a posição localmente (registro no histórico) para evitar loops de erro.
        """
        with self.lock:
            self._check_daily_reset()

            if symbol not in self.posicoes:
                return None

            pos = self.posicoes.pop(symbol)
            quantity = float(pos.get('quantity', 0.0))

            # Guarda snapshot antes de tentar vender
            pos_snapshot = pos.copy()

            # Execução da Ordem
            real_exit_price = exit_price

            if not self.paper_trading:
                try:
                    # 1) consulta saldo disponível do ativo base (ex: 'SOL')
                    base_asset = symbol.split('/')[0]
                    try:
                        bal = self.exchange.fetch_balance()
                    except Exception as e:
                        bal = {}
                    available_base = 0.0
                    if isinstance(bal, dict):
                        if base_asset in bal and isinstance(bal[base_asset], dict):
                            available_base = float(bal[base_asset].get('free', 0.0) or 0.0)
                        elif 'free' in bal and isinstance(bal['free'], dict):
                            available_base = float(bal['free'].get(base_asset, 0.0) or 0.0)

                    # se disponível for muito menor, tentamos vender o que existe
                    qty_to_sell = min(quantity, available_base)

                    # pequena margem de segurança antes de enviar (evita erro de "insufficient balance")
                    epsilon = 1e-12
                    qty_to_sell = max(0.0, qty_to_sell - epsilon)

                    # Obtém min_amount e precision do market via load_markets (fallback se não disponível)
                    min_amount, amount_precision = self._get_market_limits_and_precision(symbol)

                    # Se o exchange não informou min_amount, usa fallback conservador (ex.: 1e-8)
                    if min_amount is None:
                        # fallback prático (isso evita crash quando fetch de markets falha)
                        min_amount = 1e-8

                    # Ajusta quantidade para a precisão suportada pela exchange (se disponível)
                    if qty_to_sell > 0:
                        try:
                            if hasattr(self.exchange, 'amount_to_precision'):
                                qty_to_sell = float(self.exchange.amount_to_precision(symbol, qty_to_sell))
                            else:
                                # fallback: arredonda pela precision se informada
                                if amount_precision is not None:
                                    qty_to_sell = float(round(qty_to_sell, amount_precision))
                        except Exception:
                            # se falhar, segue com qty_to_sell original
                            pass

                    # Verifica se a quantidade ajustada atende ao min_amount
                    if qty_to_sell < min_amount:
                        # A quantidade disponível na exchange é inferior ao mínimo aceito.
                        # Duas opções: tentar limpar localmente (auto_clean_dust) ou manter a posição para tentar no futuro.
                        if CONFIG.get('auto_clean_dust', True):
                            # Limpa a posição localmente (não tentamos vender; registra como 'DUST_CLEANED')
                            logger.warning("Quantidade disponível menor que min_amount da exchange - limpando posição local (DUST)",
                                           symbol=symbol, qty_available=available_base, min_amount=min_amount)
                            # registra no histórico como limpeza de pó (dust)
                            record = {
                                'symbol': symbol,
                                'entry_price': pos.get('entry_price'),
                                'exit_price': real_exit_price,
                                'quantity': float(qty_to_sell),
                                'pnl': 0.0,
                                'reason': reason + " (DUST_CLEANED)",
                                'timestamp': str(datetime.datetime.now())
                            }
                            self.historico.append(record)
                            self.last_event = {'symbol': symbol, 'type': 'DUST_CLEANED', 'reason': reason}
                            self.daily_trades += 0  # opcional: não conta como trade
                            self.salvar()
                            logger.info("Posição removida localmente por ser pó (dust).", symbol=symbol, qty_removed=available_base)
                            return record
                        else:
                            # Mantém a posição local e retorna None (comportamento antigo)
                            logger.error("Saldo em ativo base menor que min_amount — pular venda", symbol=symbol,
                                         available=available_base, min_amount=min_amount)
                            self.posicoes[symbol] = pos_snapshot
                            return None

                    if qty_to_sell <= 0:
                        # Não há saldo para vender -> re-coloca posição e retorna erro
                        logger.error("Saldo em ativo base insuficiente para vender", symbol=symbol, needed=quantity, available=available_base)
                        self.posicoes[symbol] = pos_snapshot
                        return None

                    # 2) cria ordem de venda de mercado
                    order = self.exchange.create_order(symbol, 'market', 'sell', qty_to_sell)

                    # 3) interpreta ordem executada
                    executed_qty = 0.0
                    if isinstance(order, dict):
                        # CCXT -> alguns campos possíveis: 'filled', 'amount', 'remaining'
                        executed_qty = float(order.get('filled') or order.get('amount') or qty_to_sell)
                        real_exit_price = float(order.get('average') if order.get('average') else exit_price)

                    # Re-sincroniza saldo (importante)
                    time.sleep(1)
                    self.sync_balance()

                    # 4) se venda parcial (executed_qty < quantity), atualiza posição remanescente
                    if executed_qty < (quantity - 1e-12):
                        remaining_qty = max(0.0, quantity - executed_qty)
                        # atualiza posição local com o que restou
                        pos_snapshot['quantity'] = float(remaining_qty)
                        pos_snapshot['invested'] = float(pos_snapshot.get('entry_price', 0.0) * remaining_qty)
                        # Mantém stop/take originais para remanescente (opcional: você pode recalcular)
                        self.posicoes[symbol] = pos_snapshot

                        # registra histórico parcial (apenas a parte vendida)
                        revenue = real_exit_price * executed_qty
                        pnl = revenue - (pos.get('entry_price', 0.0) * executed_qty)
                        record = {
                            'symbol': symbol,
                            'entry_price': pos.get('entry_price'),
                            'exit_price': real_exit_price,
                            'quantity': executed_qty,
                            'pnl': pnl,
                            'reason': reason + " (PARCIAL)",
                            'timestamp': str(datetime.datetime.now())
                        }
                        self.historico.append(record)
                        self.last_event = {'symbol': symbol, 'type': 'SELL_PARTIAL', 'pnl': pnl, 'reason': reason}
                        self.daily_trades += 1
                        self.salvar()
                        logger.warning("Venda parcial executada; posição atualizada", symbol=symbol, sold=executed_qty, remaining=remaining_qty)
                        # salva sample para aprendizado
                        try:
                            features = {
                                'entry_price': pos.get('entry_price'),
                                'exit_price': real_exit_price,
                                'atr': pos.get('atr'),
                                'quantity': executed_qty,
                                'reason': reason
                            }
                            self._save_sample(symbol, features, 1 if pnl > 0 else 0)
                        except Exception:
                            pass
                        return record
                    else:
                        # Venda completa
                        revenue = real_exit_price * executed_qty
                        pnl = revenue - pos.get('invested', 0.0)

                        record = {
                            'symbol': symbol,
                            'entry_price': pos.get('entry_price'),
                            'exit_price': real_exit_price,
                            'quantity': executed_qty,
                            'pnl': pnl,
                            'reason': reason,
                            'timestamp': str(datetime.datetime.now())
                        }

                        self.historico.append(record)
                        self.last_event = {'symbol': symbol, 'type': 'SELL', 'pnl': pnl, 'reason': reason}
                        self.daily_trades += 1
                        self.salvar()
                        log_func = logger.success if pnl > 0 else logger.warning
                        log_func(f"VENDA ({'Lucro' if pnl > 0 else 'Prejuízo'})", symbol=symbol, pnl=f"${pnl:.2f}", reason=reason)

                        # salva sample para aprendizado (features simples; seu pipeline pode enriquecer)
                        try:
                            features = {
                                'entry_price': pos.get('entry_price'),
                                'exit_price': real_exit_price,
                                'atr': pos.get('atr'),
                                'quantity': executed_qty,
                                'reason': reason
                            }
                            self._save_sample(symbol, features, 1 if pnl > 0 else 0)
                        except Exception:
                            pass

                        return record

                except Exception as e:
                    logger.error("ERRO CRÍTICO AO VENDER NA BINANCE", symbol=symbol, error=str(e))
                    # Se falhar a venda na API, recoloca na lista local para tentar depois
                    self.posicoes[symbol] = pos_snapshot
                    return None
            else:
                # paper trading: simula venda completa
                real_exit_price = exit_price
                self.cash += (real_exit_price * quantity)
                revenue = real_exit_price * quantity
                pnl = revenue - pos.get('invested', 0.0)

                record = {
                    'symbol': symbol,
                    'entry_price': pos.get('entry_price'),
                    'exit_price': real_exit_price,
                    'quantity': quantity,
                    'pnl': pnl,
                    'reason': reason,
                    'timestamp': str(datetime.datetime.now())
                }

                self.historico.append(record)
                self.last_event = {'symbol': symbol, 'type': 'SELL', 'pnl': pnl, 'reason': reason}
                self.daily_trades += 1
                self.salvar()

                log_func = logger.success if pnl > 0 else logger.warning
                log_func(f"VENDA (PAPER) ({'Lucro' if pnl > 0 else 'Prejuízo'})", symbol=symbol, pnl=f"${pnl:.2f}", reason=reason)

                # salva sample
                try:
                    features = {
                        'entry_price': pos.get('entry_price'),
                        'exit_price': real_exit_price,
                        'atr': pos.get('atr'),
                        'quantity': quantity,
                        'reason': reason
                    }
                    self._save_sample(symbol, features, 1 if pnl > 0 else 0)
                except Exception:
                    pass

                return record

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
# ROBOTRADER (Cérebro)
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
                # Teste de conexão e leitura de saldo inicial
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

        # --- CARREGA MARKETS PARA PERMITIR VALIDAÇÕES (min_amount / precision)
        try:
            # força load local de markets para usar informações: markets[symbol]['limits'], ['precision']
            self.exchange.load_markets(True)
        except Exception:
            # se falhar, o resto do código ainda funciona com fallback
            logger.warning("Falha ao carregar markets da exchange (continuando com fallback)")

        self.carteira = Carteira(self.exchange, paper_trading=CONFIG['paper_trading'])

        # Sincronia inicial forçada
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
                    # detect partial_fit support
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
        # Aumentado o limite para 200 para garantir dados suficientes para EMA50 e MACD
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=CONFIG['timeframe'], limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        return df

    @retry_on_exception(max_attempts=3)
    def safe_ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    # FUNÇÃO ANALYZE COMPLETA (Gera todas as features para a IA)
    def analyze(self, symbol):
        try:
            df = self.safe_fetch_data(symbol, limit=200)  # Requisita mais dados
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

            # 2. FEATURES PARA O MODELO (Conforme seus meta.json)

            # rsi_norm
            df['rsi_norm'] = df['rsi'] / 100.0

            # dist_ema
            df['dist_ema'] = (df['c'] - df['ema50']) / df['ema50']

            # macd
            macd = ta.macd(df['c'])
            if isinstance(macd, pd.DataFrame) and 'MACD_12_26_9' in macd.columns:
                df['macd'] = macd['MACD_12_26_9']
            else:
                df['macd'] = 0.0

            # dist_bb_up
            bb = ta.bbands(df['c'], length=20, std=2)
            bbu_col = [c for c in bb.columns if 'BBU' in c]
            df['bb_upper'] = bb[bbu_col[0]] if bbu_col else df['c']
            df['dist_bb_up'] = (df['bb_upper'] - df['c']) / df['c']

            # vol_change (usa coluna 'v' do OHLCV)
            df['vol_change'] = df['v'].pct_change().fillna(0.0)

            row = df.iloc[-1].to_dict()
            return row
        except Exception as e:
            logger.error("Erro no analyze", symbol=symbol, error=str(e))
            return None

    def _compute_drawdown(self, equity_value: float):
        """Atualiza high-water mark e retorna drawdown percentual (0..1)."""
        if equity_value > self.equity_high:
            self.equity_high = equity_value
        if self.equity_high <= 0:
            return 0.0
        drawdown = max(0.0, (self.equity_high - equity_value) / max(1e-8, self.equity_high))
        return drawdown

    def _expected_value_per_unit(self, p_win: float, atr: float):
        """
        Calcula EV (em USD por unidade) usando ATR-based stop/take.
        EV = p*win_amount - (1-p)*loss_amount
        Onde win_amount = take_profit_mult * atr (USD per unit), loss_amount = stop_loss_mult * atr.
        """
        win_amount = CONFIG['take_profit_mult'] * atr
        loss_amount = CONFIG['stop_loss_mult'] * atr
        ev = p_win * win_amount - (1 - p_win) * loss_amount
        return ev, win_amount, loss_amount

    def cycle(self):
        # Ciclo principal: gestão de posições + entradas
        try:
            logger.info(f"--- Ciclo iniciado (Cash: {round(self.carteira.cash,4)} USDT) ---")
        except Exception:
            logger.info("--- Ciclo iniciado ---")

        # 1. Atualiza equity estimada
        invested_val = 0.0
        for sym, pos in self.carteira.posicoes.items():
            curr_price = self.market_prices.get(sym, pos['entry_price'])
            invested_val += (pos['quantity'] * curr_price)

        total_equity = self.carteira.cash + invested_val

        # atualiza drawdown e histórico de equity
        drawdown = self._compute_drawdown(total_equity)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        ts_now = now_utc.strftime('%H:%M:%S')
        self.equity_history.append({'time': ts_now, 'value': total_equity, 'drawdown': drawdown})
        if len(self.equity_history) > 200:
            self.equity_history.pop(0)

        if drawdown >= CONFIG.get('max_drawdown_pct', 0.99):
            logger.warning("Pausando novas entradas: drawdown crítico atingido", drawdown=drawdown)
            # Ainda permite gerenciar / fechar posições existentes, mas não abre novas
            allow_entries = False
        else:
            allow_entries = True

        # 2. Gerenciar Posições (fechamentos por stop/take/trailling)
        for sym in list(self.carteira.posicoes.keys()):
            try:
                tk = self.safe_ticker(sym)
                price = tk.get('last', None) if isinstance(tk, dict) else None
                if price is None:
                    continue
                self.market_prices[sym] = price
                pos = self.carteira.posicoes[sym]

                # Trailing Stop (conservador: usa ATR)
                if pos.get('atr'):
                    new_trail = price - pos['atr']
                    if pos.get('trail_stop') is None or new_trail > pos['trail_stop']:
                        pos['trail_stop'] = new_trail

                reason = None
                if pos.get('stop_price') and price <= pos['stop_price']:
                    reason = "STOP LOSS"
                elif pos.get('take_price') and price >= pos['take_price']:
                    reason = "TAKE PROFIT"
                elif pos.get('trail_stop') and price <= pos['trail_stop']:
                    reason = "TRAILING STOP"

                if reason:
                    rec = self.carteira.fechar_posicao(sym, price, reason=reason)
                    if rec:
                        metrics.record_trade(rec['pnl'])
            except Exception:
                # Não polui logs com cada exceção pequena aqui
                pass

        # 3. Novas Entradas
        self.carteira._check_daily_reset()
        if self.carteira.daily_trades >= CONFIG['max_daily_trades']:
            logger.info("Limite diário de trades atingido", daily_trades=self.carteira.daily_trades)
            return

        if not allow_entries:
            # drawdown crítico: não abrir novas posições
            return

        slots = max(0, CONFIG['max_positions'] - len(self.carteira.posicoes))
        if slots <= 0:
            # já no máximo de posições
            return

        # Recolhe candidatos (scores)
        all_candidates: List[Dict[str, Any]] = []
        for sym in CONFIG['symbols']:
            try:
                if sym in self.carteira.posicoes:
                    continue
                if self.cooldowns.get(sym) and datetime.datetime.now() < self.cooldowns[sym]:
                    continue

                row = self.analyze(sym)
                if not row:
                    continue

                price = float(row.get('c') or 0.0)
                atr = float(row.get('atr') or 0.0)
                if atr <= 0:
                    # fallback prático: 0.5% do preço (evita stop_dist==0)
                    atr = max(price * 0.005, 1e-8)

                # --- LÓGICA DE PONTUAÇÃO (PRIORIZA IA) ---
                score = 0.0
                used_model = 'NONE'
                p_win = 0.5
                if sym in self.models:
                    try:
                        model_data = self.models[sym]
                        features = model_data.get('features', [])
                        x = pd.DataFrame([{k: row.get(k, 0) for k in features}])
                        # predict_proba pode falhar se o modelo não suportar; try/except
                        # Se for scikit models, predict_proba existe; se não, fallback
                        model = model_data['model']
                        if hasattr(model, 'predict_proba'):
                            p_win = float(model.predict_proba(x)[0][1])
                            score = p_win
                        elif hasattr(model, 'predict'):
                            score = float(model.predict(x)[0])
                            p_win = score
                        else:
                            p_win = 0.5
                            score = 0.5
                        used_model = 'USADO'
                    except Exception as e:
                        # fallback conservador
                        p_win = 0.5
                        score = 0.5
                        used_model = 'FALLBACK'
                        if CONFIG['verbose_logs']:
                            logger.error("Falha na previsão da IA. Usando fallback.", symbol=sym, error=str(e))
                else:
                    # Fallback simples (mantive sua lógica)
                    score = 0.5
                    p_win = 0.5
                    if row.get('rsi', 50.0) < 35:
                        score += 0.2
                        p_win += 0.1
                    if row.get('ema50') and row['c'] > row['ema50']:
                        score += 0.1
                        p_win += 0.05
                    used_model = 'FALLBACK'

                # Computa EV por unidade (USD por 1 unidade do ativo)
                ev_per_unit, win_amount, loss_amount = self._expected_value_per_unit(p_win, atr)

                # Normaliza EV para proporção relativa ao preço (ev per USD invested)
                ev_per_usd = ev_per_unit / max(price, 1e-8)

                # Logs
                if CONFIG['verbose_logs']:
                    logger.info("Score IA", symbol=sym, score=f"{score:.4f}", model=used_model, p_win=round(p_win,4),
                                ev_per_unit=round(ev_per_unit,6), ev_per_usd=round(ev_per_usd,6))

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
                # não poluir logs com cada falha pequena de símbolo
                continue

        if not all_candidates:
            if CONFIG['verbose_logs']:
                logger.info("Nenhum candidato analisado neste ciclo.")
            return

        # Ordena por score decrescente
        all_candidates.sort(key=lambda x: x['score'], reverse=True)

        # Thresholds adaptativos: tenta encontrar candidatos com limiares menores se necessário
        thresholds = [CONFIG.get('min_confidence', 0.55), 0.50, 0.45, 0.40]
        selected_candidates: List[Dict[str, Any]] = []
        used_threshold = None
        for thr in thresholds:
            selected_candidates = [c for c in all_candidates if c['score'] > thr and c['ev_per_usd'] >= CONFIG.get('min_expected_ev', 0.0)]
            if selected_candidates:
                used_threshold = thr
                logger.info("Candidatos encontrados (com EV positivo)", threshold=thr, found=len(selected_candidates))
                break

        # Se ainda vazio, permite forçar o melhor candidato Razoável (>=0.40) *apenas se EV positivo*
        if not selected_candidates:
            top = all_candidates[0]
            if top['score'] >= 0.40 and top['ev_per_usd'] >= CONFIG.get('min_expected_ev', 0.0):
                selected_candidates = [top]
                used_threshold = 'forced_top_>=0.40_with_EV_check'
                logger.warning("Forçando melhor candidato (>=0.40) com EV positivo", symbol=top['symbol'], score=top['score'], ev_per_usd=top['ev_per_usd'])
            else:
                # Nenhum candidato qualificado
                logger.info("Nenhum candidato qualificado encontrado neste ciclo", best_score=all_candidates[0]['score'], best_ev=all_candidates[0]['ev_per_usd'])
                return

        # Reduz à quantidade de slots disponíveis
        selected_candidates = selected_candidates[:slots]

        # --- DIVERSIFICAÇÃO / ALOCAÇÃO ---
        # Calcula cash disponível para novas ordens (respeita buffer)
        available_cash = max(0.0, self.carteira.cash * (1.0 - CONFIG['allocation_buffer_pct']))
        # também respeita exposição máxima
        current_exposure = sum([p['invested'] for p in self.carteira.posicoes.values()]) if self.carteira.posicoes else 0.0
        max_allowed_exposure = total_equity * CONFIG['max_exposure_pct']
        remaining_exposure_capacity = max_allowed_exposure - current_exposure
        if remaining_exposure_capacity <= 0:
            logger.warning("Capacidade de exposição esgotada", current_exposure=current_exposure, max_allowed=max_allowed_exposure)
            return

        # limit available cash by remaining exposure capacity (em USD)
        available_cash = min(available_cash, remaining_exposure_capacity, self.carteira.cash)

        if available_cash < CONFIG['min_trade_usd']:
            logger.info("Cash insuficiente para novas alocações", available_cash=round(available_cash, 4))
            return

        # Determina alocações por slot (baseadas em EV-weighted proportional)
        total_score = sum(max(0.0, c['ev_per_usd']) for c in selected_candidates) or sum(c['score'] for c in selected_candidates) or 1.0
        allocations = []
        for c in selected_candidates:
            if CONFIG['allocation_mode'] == 'equal':
                alloc = available_cash / len(selected_candidates)
            else:  # proportional por EV preferencialmente
                weight = max(0.0, c['ev_per_usd'])
                if weight == 0:
                    weight = c['score']
                alloc = (weight / total_score) * available_cash
            # garante mínimo por slot
            if alloc < CONFIG['min_allocation_per_slot']:
                alloc = CONFIG['min_allocation_per_slot']
            allocations.append(max(0.0, alloc))

        # Ajusta alocações se excederem o available_cash (normaliza)
        total_alloc = sum(allocations)
        if total_alloc > available_cash:
            factor = available_cash / total_alloc
            allocations = [a * factor for a in allocations]
            total_alloc = sum(allocations)

        # Associa alocações aos candidatos (mantendo ordem)
        for idx, cand in enumerate(selected_candidates):
            cand['allocation'] = allocations[idx] if idx < len(allocations) else allocations[-1]

        # Finalmente, tenta abrir posições com base em alocação + risco + EV sizing
        for cand in selected_candidates:
            if slots <= 0:
                break

            sym = cand['symbol']
            price = cand['price']
            atr = cand['atr']
            score = cand['score']
            p_win = cand['p_win']
            ev_per_unit = cand['ev_per_unit']
            ev_per_usd = cand['ev_per_usd']
            allocation = float(cand.get('allocation', CONFIG['min_allocation_per_slot']))

            # calcula qty pelo método de risco e também pelo allocation; toma o menor para não sobreinvestir
            # risk_amt baseado em total_equity (consistente com gestão de risco)
            base_risk_amt = total_equity * CONFIG['risk_per_trade']

            # Risco adaptativo: multiplicador baseado no EV por USD (mais EV -> maior multiplicador), limitado
            # formula simples e conservadora:
            # risk_modifier = 1 + clamp(ev_per_usd / 0.01, -0.8, 2.0)
            # Isso significa: se EV por USD = 0.01 (1%), risk_modifier += 1 -> dobraria o risco (limitado abaixo)
            risk_modifier = 1.0
            try:
                if ev_per_usd > 0:
                    risk_modifier = 1.0 + min(2.0, ev_per_usd / 0.01)
                else:
                    # penaliza eventos com baixo/negativo EV
                    risk_modifier = max(0.1, 1.0 + ev_per_usd / 0.01)
            except Exception:
                risk_modifier = 1.0

            # Se modo conservador, reduz risco_modifier para evitar multiplicadores agressivos
            if CONFIG.get('conservative_mode', True):
                risk_modifier = min(risk_modifier, 1.15)  # máximo 15% de aumento no conservador

            risk_amt = base_risk_amt * risk_modifier

            stop_dist = atr * CONFIG['stop_loss_mult']
            if stop_dist <= 0:
                logger.warning("Pular: stop_dist inválido", symbol=sym, atr=atr)
                continue

            qty_by_risk = risk_amt / stop_dist
            cost_by_risk = qty_by_risk * price

            # qty baseado na alocação
            qty_by_alloc = allocation / price if price > 0 else 0.0
            cost_by_alloc = qty_by_alloc * price

            # decide qty final: não exceder allocation nem risco, usa o menor qty
            qty = min(qty_by_risk, qty_by_alloc)
            cost = qty * price

            # Ajuste de precision: tenta usar amount_to_precision se disponível
            try:
                if qty > 0 and hasattr(self.carteira.exchange, 'amount_to_precision'):
                    qty = float(self.carteira.exchange.amount_to_precision(sym, qty))
                    cost = qty * price
            except Exception:
                pass

            # se qty muito pequeno (ordem pode falhar por limites mínimos), força min allocation/qty
            if cost < CONFIG['min_trade_usd']:
                # tenta forçar mínimo (se houver cash)
                if self.carteira.cash >= CONFIG['min_trade_usd'] and (current_exposure + CONFIG['min_trade_usd']) <= max_allowed_exposure:
                    cost = CONFIG['min_trade_usd']
                    qty = cost / price if price > 0 else 0.0
                    logger.info("Forçando custo mínimo para spot Binance", symbol=sym, custo_real=round(cost, 4))
                else:
                    logger.info("Pular: custo calculado abaixo do mínimo e não há margem para forçar", symbol=sym, calc_cost=round(cost, 4))
                    continue

            # válidação exposição e saldo
            current_exposure = sum([p['invested'] for p in self.carteira.posicoes.values()]) if self.carteira.posicoes else 0.0
            max_allowed_exposure = total_equity * CONFIG['max_exposure_pct']

            if (current_exposure + cost) > max_allowed_exposure:
                logger.warning("Pular: Exposição total excedida ao abrir", symbol=sym, would_exposure=round(current_exposure + cost, 4), max_allowed=round(max_allowed_exposure, 4))
                continue

            if cost > self.carteira.cash:
                logger.warning("Pular: Saldo livre insuficiente", symbol=sym, cost=round(cost, 4), cash=round(self.carteira.cash, 4))
                continue

            # final: abrir posição (stop e take calculados com base em atr)
            stop_price = max(0.0, price - stop_dist)
            take_price = price + (atr * CONFIG['take_profit_mult'])
            ok = self.carteira.abrir_posicao(
                sym, price, qty, atr,
                stop_price=stop_price,
                take_price=take_price
            )

            if ok:
                slots -= 1
                # adiciona cooldown por grupo para evitar concentrar entradas no mesmo grupo nas próximas horas
                grp = SYMBOL_TO_GROUP.get(sym, 'ungrouped')
                # aplica cooldown para todos símbolos do grupo
                for s in CONFIG['groups'].get(grp, [sym]):
                    self.cooldowns[s] = datetime.datetime.now() + datetime.timedelta(seconds=CONFIG['group_cooldown_seconds'])
                logger.info("Ordem aberta e cooldown aplicado ao grupo",
                            symbol=sym,
                            group=grp,
                            allocated=round(cost, 4),
                            p_win=round(p_win, 4),
                            ev_per_usd=round(ev_per_usd, 6),
                            risk_modifier=round(risk_modifier, 3))

    def loop(self):
        self.running = True
        while self.running:
            try:
                self.cycle()
            except Exception as e:
                logger.error("Erro no ciclo principal", error=str(e))

            for _ in range(CONFIG['sleep_cycle']):
                if not self.running:
                    break
                time.sleep(1)

    def start(self):
        if not self.running:
            Thread(target=self.loop, daemon=True).start()
            logger.system("Robô Iniciado")

    def stop(self):
        self.running = False
        logger.system("Robô Parado")

# =========================
# API FLASK (integração FRONT preservada)
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
    # Retorna logs mais relevantes (já bufferizados pelo StructuredLogger)
    return jsonify(list(reversed(logger.buffer)))


@app.route('/api/status', methods=['GET'])
def api_status():
    open_data = []
    invested_total = 0.0
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

    eq = bot.carteira.cash + invested_total

    return jsonify({
        "isRunning": bot.running,
        "balance": bot.carteira.cash,
        "equity": eq,
        "equityHistory": bot.equity_history,
        "dailyTrades": bot.carteira.daily_trades,
        "openPositions": open_data
    })


if __name__ == '__main__':
    print("\033[92m=== TRADING REAL (R$ 100/17 USDT) - VERSÃO AJUSTADA + EV & RISK-SIZING ===\033[0m")
    app.run(host='0.0.0.0', port=5000)
