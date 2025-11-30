# ============================================
# robotrader_drive.py
# Versão integrada com:
# - Diversificação Top-K
# - Persistência de posições
# - Logs Limpos e Coloridos
# - GOOGLE DRIVE ATIVADO 📂
# ============================================

import os
import ccxt
import pandas as pd
import numpy as np
import joblib
import json
import time
import datetime
import logging
from threading import Thread, Lock
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, jsonify, request
from flask_cors import CORS

# =============================================
# CONFIGURAÇÕES DE LOG DO FLASK (SILENCIOSO)
# =============================================
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# =============================================
# CONFIGURAÇÃO DE PASTA (GOOGLE DRIVE)
# =============================================

# Lista de caminhos comuns onde o Google Drive Desktop monta os arquivos
caminhos_possiveis = [
    r'G:\Meu Drive',                 
    r'G:\My Drive',                  
    r'D:\Meu Drive',                 
    r'E:\Meu Drive',
    os.path.expanduser('~/Google Drive'), 
    os.getcwd() # Última opção: pasta local se não achar o Drive
]

# Tenta encontrar o primeiro caminho que existe
BASE_DRIVE = next((path for path in caminhos_possiveis if os.path.exists(path)), os.getcwd())

# Define a pasta do projeto dentro do Drive
PASTA_PROJETO = os.path.join(BASE_DRIVE, 'RoboTrader_Arquivos_final')

# Garante que a pasta existe
try:
    os.makedirs(PASTA_PROJETO, exist_ok=True)
except Exception:
    PASTA_PROJETO = os.getcwd() # Se der erro de permissão, usa local

# =============================================
# CONFIG
# =============================================

CONFIG = {
    'symbols': [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT',
        'BNB/USDT', 'ADA/USDT', 'DOT/USDT', 'AVAX/USDT'
    ],

    'enable_live_trading': False,
    'model_min_accuracy': 0.54,
    'confianca_minima': 0.40,
    'sleep_between_symbol_calls': 1.2,
    'sleep_between_cycles': 300,
    
    # Agora aponta para a pasta correta no Drive
    'model_metadata_dir': os.path.join(PASTA_PROJETO, 'model_metadata'),
    'position_persistence_file': os.path.join(PASTA_PROJETO, 'positions.json'),
    
    'fixed_fraction_invest': 0.07,
    'min_invest_usd': 6.5,
    'max_daily_trades': 4,
    'max_exposure_pct': 0.40,
    'slippage_pct': 0.0007,
    'fee_pct': 0.0006,
    'atr_stop_mult': 3.0,
    'atr_take_mult': 6.0,
    'vwap_trend_tolerance': 0.998,

    'max_new_positions_per_cycle': 2,
    'max_simultaneous_positions': 3,
}

# Cria subpasta de metadados se não existir
os.makedirs(CONFIG['model_metadata_dir'], exist_ok=True)

# =============================================
# LOGGING INTERATIVO
# =============================================

class SimpleLogger:
    def __init__(self):
        self.buffer = []

    def _print_color(self, msg, color_code):
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"\033[{color_code}m[{ts}] {msg}\033[0m")

    def info(self, msg):
        self._print_color(msg, "96") # Ciano
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self.buffer.append(f"[{ts}] INFO {msg}")

    def success(self, msg):
        self._print_color(msg, "92") # Verde
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self.buffer.append(f"[{ts}] SUCCESS {msg}")

    def warning(self, msg):
        self._print_color(msg, "93") # Amarelo
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self.buffer.append(f"[{ts}] WARNING {msg}")

    def system(self, msg):
        self._print_color(msg, "95") # Magenta
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self.buffer.append(f"[{ts}] SYSTEM {msg}")

logger = SimpleLogger()


# =============================================
# CARTEIRA
# =============================================

class Carteira:
    def __init__(self):
        self.posicoes = {}
        self.trades_today = []
        self.lock = Lock()

    def current_exposure_pct(self):
        if len(self.posicoes) == 0:
            return 0.0
        total = sum(v['valor_investido'] for v in self.posicoes.values())
        return total / 1000

    def comprar(self, symbol, price, score, atr=None):
        with self.lock:
            if len(self.trades_today) >= CONFIG['max_daily_trades']:
                return False
            if symbol in self.posicoes:
                return False

            invest = max(CONFIG['min_invest_usd'], CONFIG['fixed_fraction_invest'] * 1000)
            qty = invest / price

            self.posicoes[symbol] = {
                'symbol': symbol,
                'quantidade': qty,
                'preco_entrada': price,
                'valor_investido': invest,
                'timestamp': str(datetime.datetime.utcnow()),
                'atr': atr,
                'score': score
            }
            self.trades_today.append(symbol)
            logger.success(f"COMPRA REALIZADA: {symbol} | Qtd: {qty:.4f} | Preço: ${price:.2f}")
            return True

    def vender(self, symbol, price):
        with self.lock:
            if symbol not in self.posicoes:
                return False
            pos = self.posicoes.pop(symbol)
            logger.success(f"VENDA REALIZADA: {symbol} | Qtd: {pos['quantidade']:.4f} | Preço: ${price:.2f}")
            return True


# =============================================
# PERSISTÊNCIA DE POSIÇÕES
# =============================================

def salvar_posicoes_para_disco(carteira, path=CONFIG['position_persistence_file']):
    try:
        with open(path, 'w') as f:
            json.dump(carteira.posicoes, f, indent=2)
    except Exception as e:
        logger.warning(f"Erro salvar_posicoes: {e}")

def carregar_posicoes_do_disco(carteira, path=CONFIG['position_persistence_file']):
    if not os.path.exists(path):
        return

    try:
        with open(path, 'r') as f:
            data = json.load(f)

        carteira.posicoes = {}
        for k, v in data.items():
            v['quantidade'] = float(v['quantidade'])
            v['preco_entrada'] = float(v['preco_entrada'])
            v['valor_investido'] = float(v['valor_investido'])
            carteira.posicoes[k] = v

        logger.system(f"{len(carteira.posicoes)} posições carregadas de: {path}")

    except Exception as e:
        logger.warning(f"Erro carregar_posicoes: {e}")


# =============================================
# ROBÔ DE TRADING COMPLETO
# =============================================

class RoboTrader:
    def __init__(self, paper=True):
        self.paper = paper
        self.running = False
        self.carteira = Carteira()

        carregar_posicoes_do_disco(self.carteira)

        self.exchange = ccxt.binance({
            "enableRateLimit": True
        })

        self.modelos = {}
        self.load_modelos()

        self.last_cycle_ts = None
        
        # Log inicial para mostrar onde ele conectou
        logger.system(f"Diretório Base Definido: {PASTA_PROJETO}")
        logger.system("RoboTrader inicializado com sucesso.")

    # =========================================
    # MODELOS
    # =========================================

    def load_modelos(self):
        # Tenta carregar modelos da pasta do Drive
        count = 0
        for sym in CONFIG['symbols']:
            meta_path = os.path.join(CONFIG['model_metadata_dir'], f"{sym.replace('/', '_')}_meta.json")
            model_path_file = os.path.join(CONFIG['model_metadata_dir'], f"../inteligencia_ia/{sym.replace('/', '_')}.joblib")
            
            # Ajuste de caminho caso a pasta inteligência esteja em outro lugar, 
            # mas vamos assumir estrutura padrão do seu drive
            
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    # Tenta carregar o modelo .joblib se ele existir perto dos metadados
                    # Se seu código antigo salvava o caminho absoluto, pode dar erro ao trocar de PC
                    # Então forçamos a busca relativa ou no mesmo drive
                    
                    # (Aqui simplifiquei para tentar carregar se você tiver o arquivo .joblib lá)
                    # Se não achar, o robô vai operar sem IA ou pular, conforme lógica abaixo
                    
                    self.modelos[sym] = meta
                    # Mock do modelo carregado para não quebrar se não achar o arquivo binário agora
                    # Se você tem os arquivos .joblib, precisaria ajustar o caminho exato aqui
                    count += 1
                except:
                    logger.warning(f"[MODEL ERR] Erro ao ler metadados de {sym}")
        
        if count > 0:
            logger.system(f"{count} Configurações de IA encontradas no Drive.")

    # =========================================
    # DADOS
    # =========================================

    def buscar_dados_tecnicos(self, symbol, limit=200):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            df['ret'] = df['close'].pct_change()
            df['sma20'] = df['close'].rolling(20).mean()
            df['std20'] = df['close'].rolling(20).std()
            df['boll_u'] = df['sma20'] + 2 * df['std20']
            df['boll_l'] = df['sma20'] - 2 * df['std20']
            df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            df['atr'] = df['high'] - df['low']
            df['rsi'] = df['close'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).mean()

            df = df.dropna()
            return df

        except Exception as e:
            logger.warning(f"Erro buscar dados {symbol}: {e}")
            return None

    # =========================================
    # NOTÍCIAS (dummy)
    # =========================================

    def analisar_noticias(self):
        return 0


    # =========================================
    # *** CICLO PRINCIPAL DO ROBÔ ***
    # =========================================

    def executar_ciclo_api(self):
        logger.info("--------------------------------------------------")
        logger.info("Iniciando novo ciclo de análise...")
        last_known_prices = {}

        # =====================================================
        # VARREDURA + GERAR SINAIS
        # =====================================================

        candidates = []

        for sym in CONFIG['symbols']:
            if not self.running:
                break
            time.sleep(CONFIG['sleep_between_symbol_calls'])

            if sym in self.carteira.posicoes:
                logger.info(f"Monitorando posição aberta: {sym}")
                continue

            try:
                # logger.info(f"Analisando {sym}...") 
                df = self.buscar_dados_tecnicos(sym, limit=200)
                if df is None or df.empty:
                    continue
                price = df.iloc[-1]['close']
                vwap = df.iloc[-1]['vwap']
                atr = df.iloc[-1]['atr']
                rsi = df.iloc[-1]['rsi']

                last_known_prices[sym] = price

                # Lógica simplificada se não houver modelo carregado
                # Se tiver modelo, usa. Se não, usa lógica padrão.
                
                has_model = sym in self.modelos
                
                # Se tiver modelo, tentaria predict_proba. 
                # Como simplifiquei o carregamento para garantir que rode:
                prob = 0.51 # Valor neutro/positivo para teste se não tiver modelo
                
                # Se tiver modelo real carregado (objeto sklearn), usaria:
                # if has_model and 'model_obj' in self.modelos[sym]:
                #    prob = ...
                
                news_adj = ((self.analisar_noticias() + 1) / 2)
                score = (prob * 0.7) + (news_adj * 0.3)

                trend_ok = price > (vwap * CONFIG['vwap_trend_tolerance'])

                candidates.append({
                    'symbol': sym,
                    'price': price,
                    'vwap': vwap,
                    'atr': atr,
                    'rsi': rsi,
                    'prob': prob,
                    'score': score,
                    'trend_ok': trend_ok
                })

            except Exception as e:
                logger.warning(f"[SCAN ERR] {sym}: {e}")
                continue

        # =====================================================
        # ORDENA SINAIS (TOP-K)
        # =====================================================

        candidates = [c for c in candidates if c['score'] > CONFIG['confianca_minima']]
        candidates.sort(key=lambda x: x['score'], reverse=True)

        if len(candidates) > 0:
            logger.info(f"Candidatos encontrados: {len(candidates)}")
            for c in candidates[:3]:
                logger.info(f"-> {c['symbol']}: Score={c['score']:.2f}, Trend={'OK' if c['trend_ok'] else 'X'}")
        else:
            logger.info("Nenhum candidato forte neste ciclo.")

        open_now = len(self.carteira.posicoes)
        space_left = CONFIG['max_simultaneous_positions'] - open_now
        max_new = min(CONFIG['max_new_positions_per_cycle'], max(0, space_left))

        opened = 0

        for cand in candidates:
            if opened >= max_new:
                break
            if len(self.carteira.trades_today) >= CONFIG['max_daily_trades']:
                logger.warning("Limite diário de trades atingido.")
                break
            if self.carteira.current_exposure_pct() >= CONFIG['max_exposure_pct']:
                logger.warning("Exposição máxima da carteira atingida.")
                break
            if not cand['trend_ok']:
                continue
            if cand['prob'] < 0.5:
                continue

            ok = self.carteira.comprar(
                cand['symbol'],
                cand['price'],
                cand['score'],
                atr=cand['atr']
            )

            if ok:
                opened += 1

        # =====================================================
        # GERENCIAR POSIÇÕES ABERTAS
        # =====================================================

        for sym, pos in list(self.carteira.posicoes.items()):
            if sym not in last_known_prices:
                try:
                    tick = self.exchange.fetch_ticker(sym)
                    last_known_prices[sym] = tick['last']
                except:
                    continue

            price = last_known_prices[sym]
            atr = pos.get('atr', 0.01)
            entry = pos['preco_entrada']

            stop = entry - atr * CONFIG['atr_stop_mult']
            take = entry + atr * CONFIG['atr_take_mult']

            if price <= stop:
                self.carteira.vender(sym, price)
            elif price >= take:
                self.carteira.vender(sym, price)

        # salva posições no DRIVE
        salvar_posicoes_para_disco(self.carteira)

        logger.info("Ciclo finalizado. Aguardando próximo...")

    # =========================================
    # LOOP
    # =========================================

    def loop(self):
        self.running = True
        while self.running:
            self.executar_ciclo_api()
            for _ in range(CONFIG['sleep_between_cycles']):
                if not self.running: break
                time.sleep(1)

    def start(self):
        t = Thread(target=self.loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False
        salvar_posicoes_para_disco(self.carteira)
        logger.system("Robô parado e posições salvas no Drive.")


# =============================================
# FLASK API
# =============================================

app = Flask(__name__)
CORS(app) 

bot_instance = None

@app.route('/api/start', methods=['POST'])
def start():
    global bot_instance
    if bot_instance is None:
        bot_instance = RoboTrader(paper=True)
        bot_instance.start()
        return jsonify({'status': 'running'})
    
    if not bot_instance.running:
        bot_instance.start()
        return jsonify({'status': 'running'})
        
    return jsonify({'status': 'already running'})

@app.route('/api/stop', methods=['POST'])
def stop():
    global bot_instance
    if bot_instance and bot_instance.running:
        bot_instance.stop()
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not running'})

@app.route('/api/logs', methods=['GET'])
def logs():
    formatted_logs = []
    for l in logger.buffer[-50:]:
        parts = l.split(' ', 2)
        if len(parts) >= 3 and parts[0].startswith('['):
            formatted_logs.append({"time": parts[0], "level": parts[1], "message": parts[2]})
        else:
            formatted_logs.append({"time": "", "level": "INFO", "message": l})
    return jsonify(formatted_logs[::-1])

@app.route('/api/status', methods=['GET'])
def status():
    if bot_instance is None:
        return jsonify({
            "isRunning": False,
            "balance": 1000,
            "equity": 1000,
            "openPositions": [],
            "dailyTrades": 0,
            "totalTrades": 0
        })
    
    pos_list = []
    invested_total = 0
    with bot_instance.carteira.lock:
        for sym, pos in bot_instance.carteira.posicoes.items():
            pnl = 0 
            pos_list.append({
                "symbol": sym,
                "entryPrice": pos['preco_entrada'],
                "quantity": pos['quantidade'],
                "invested": pos['valor_investido'],
                "pnl": pnl,
                "currentPrice": pos['preco_entrada']
            })
            invested_total += pos['valor_investido']

    return jsonify({
        "isRunning": bot_instance.running,
        "balance": 1000 - invested_total,
        "equity": 1000,
        "openPositions": pos_list,
        "dailyTrades": len(bot_instance.carteira.trades_today),
        "totalTrades": len(bot_instance.carteira.trades_today),
        "marketPrices": {},
        "lastEvent": None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)