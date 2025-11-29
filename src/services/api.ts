import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export const api = {
  getStatus: async () => {
    try {
      const response = await axios.get(`${API_URL}/status`);
      return response.data;
    } catch (error) {
      console.error("Erro API:", error);
      return null;
    }
  },
  getLogs: async () => {
    try {
      const response = await axios.get(`${API_URL}/logs`);
      return response.data;
    } catch { return []; }
  },
  getHistory: async () => {
    try {
      const response = await axios.get(`${API_URL}/history`);
      return response.data;
    } catch { return []; }
  },
  startBot: () => axios.post(`${API_URL}/start`),
  stopBot: () => axios.post(`${API_URL}/stop`)
};