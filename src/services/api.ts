import axios from 'axios';

// Aponta para a porta do Flask
const API_URL = 'http://localhost:5000/api';

// --- INTERFACES EXPORTADAS (Para usar no Dashboard) ---
export interface Position {
  symbol: string;
  entryPrice: number;
  quantity: number;
  invested: number;
  pnl: number;
  currentPrice?: number;
}

export interface BotStatus {
  isRunning: boolean;
  balance: number;
  equity: number;
  dailyTrades: number;
  totalTrades: number;
  openPositions: Position[];
  marketPrices: Record<string, number>; // Obrigatório para o Ticker
  lastEvent: { symbol: string, type: 'BUY' | 'SELL', price?: number, pnl?: number } | null;
}

export interface TradeHistoryItem {
  timestamp: string;
  symbol: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  reason?: string;
}
// -------------------------------------------------------

export const api = {
  // Métodos de Controle
  start: () => axios.post(`${API_URL}/start`),
  stop: () => axios.post(`${API_URL}/stop`),
  
  // Aliases que o Dashboard usa
  startBot: () => axios.post(`${API_URL}/start`),
  stopBot: () => axios.post(`${API_URL}/stop`),

  // Métodos de Dados (Tipados)
  getStatus: async (): Promise<BotStatus | null> => {
    try {
      const response = await axios.get(`${API_URL}/status`);
      return response.data;
    } catch (error) {
      console.error("Erro API Status:", error);
      return null;
    }
  },
  getLogs: async () => {
    try {
      const response = await axios.get(`${API_URL}/logs`);
      return response.data;
    } catch { return []; }
  },
  getHistory: async (): Promise<TradeHistoryItem[]> => {
    try {
      const response = await axios.get(`${API_URL}/history`);
      return response.data;
    } catch { return []; }
  },

  // --- FUNÇÃO: VENDA MANUAL ---
  closePosition: async (symbol: string) => {
    try {
      const response = await axios.post(`${API_URL}/close_position`, { symbol });
      return response.data;
    } catch (error) {
      console.error("Erro ao fechar posição:", error);
      throw error;
    }
  },

  // --- NOVA FUNÇÃO: COMPRA MANUAL ---
  // Chama o endpoint /api/open_position criado no app.py
  openPosition: async (symbol: string, amount: number) => {
    try {
      const response = await axios.post(`${API_URL}/open_position`, { symbol, amount });
      return response.data;
    } catch (error) {
      console.error("Erro ao abrir posição:", error);
      throw error;
    }
  }
};