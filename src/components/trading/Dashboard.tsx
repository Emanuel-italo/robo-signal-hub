import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Wallet, DollarSign, Activity, Power, TrendingUp, Target, BrainCircuit, Zap, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from "recharts";
import { api } from "@/services/api";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { useWakeLock } from "@/hooks/use-wake-lock";
import MarketTicker from "./MarketTicker";

interface BotStatus {
  isRunning: boolean;
  balance: number;
  equity: number;
  dailyTrades: number;
  totalTrades: number;
  marketPrices: Record<string, number>;
  lastEvent: { symbol: string, type: 'BUY' | 'SELL', price?: number, pnl?: number } | null;
}

interface ChartData {
  time: string;
  value: number;
}

const COLORS = ['#ef4444', '#fca5a5'];

const Dashboard = () => {
  const { data: rawStatus, refetch: refetchStatus } = useQuery({
    queryKey: ["botStatus"],
    queryFn: api.getStatus,
    refetchInterval: 1000, 
  });

  const { data: history } = useQuery({
    queryKey: ["botHistory"],
    queryFn: api.getHistory,
    refetchInterval: 5000,
  });

  const status = rawStatus as BotStatus | undefined;
  const { requestLock, releaseLock } = useWakeLock();
  
  const [equityData, setEquityData] = useState<ChartData[]>([]);
  const [winRateData, setWinRateData] = useState<{name: string, value: number}[]>([]);
  const [trend, setTrend] = useState<'up' | 'down' | 'neutral'>('neutral');

  // Notificações
  useEffect(() => {
    if (status?.lastEvent) {
      const evt = status.lastEvent;
      if (evt.type === 'BUY') {
        toast.success(`COMPRA EXECUTADA: ${evt.symbol}`, {
          description: `Entrada em $${evt.price?.toFixed(2)}`,
          className: "bg-black border-2 border-red-500/50 text-white font-bold"
        });
      } else if (evt.type === 'SELL') {
        const isWin = (evt.pnl || 0) > 0;
        toast(isWin ? "LUCRO REALIZADO!" : "STOP LOSS", {
          description: `${evt.symbol} PnL: $${evt.pnl?.toFixed(2)}`,
          className: isWin ? "bg-emerald-950 border-emerald-500 text-emerald-400" : "bg-red-950 border-red-500 text-red-400"
        });
      }
    }
  }, [status?.lastEvent]);

  // Controle de Tela
  useEffect(() => {
    if (status?.isRunning) requestLock();
    else releaseLock();
  }, [status?.isRunning, requestLock, releaseLock]);

  useEffect(() => {
    if (status?.equity) {
      const now = new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      setEquityData(prev => {
        const newData = [...prev, { time: now, value: status.equity }];
        if (prev.length > 0) {
          const last = prev[prev.length - 1].value;
          setTrend(status.equity > last ? 'up' : status.equity < last ? 'down' : 'neutral');
        }
        return newData.slice(-40); 
      });
    }
  }, [status]);

  useEffect(() => {
    if (history && Array.isArray(history)) {
      const wins = history.filter((t: any) => t.pnl > 0).length;
      const losses = history.filter((t: any) => t.pnl <= 0).length;
      setWinRateData([
        { name: 'Wins', value: wins },
        { name: 'Losses', value: losses }
      ]);
    }
  }, [history]);

  const handleToggleBot = async () => {
    if (status?.isRunning) await api.stopBot();
    else await api.startBot();
    refetchStatus();
  };

  return (
    <div className="space-y-8 animate-in fade-in zoom-in-95 duration-700">

      {/* --- AQUI ESTÁ O TICKER QUE ESTAVA FALTANDO --- */}
      {status?.marketPrices && (
        <div className="rounded-xl border border-red-900/30 overflow-hidden shadow-lg bg-black/40 backdrop-blur-sm">
           <MarketTicker prices={status.marketPrices} />
        </div>
      )}
      
      {/* Banner de Controle */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-red-600 to-red-500 p-8 text-white shadow-2xl shadow-red-500/30">
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-64 h-64 bg-white/10 rounded-full blur-3xl"></div>
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-white/20 backdrop-blur-md rounded-2xl">
              <BrainCircuit className="w-8 h-8 text-white" />
            </div>
            <div>
              <h2 className="text-3xl font-black tracking-tight">PAINEL DE COMANDO</h2>
              <div className="flex items-center gap-2">
                <p className="text-red-100 font-medium opacity-90">Inteligência Artificial Operacional</p>
                {status?.isRunning && (
                    <Badge className="bg-white/20 text-white border-none animate-pulse">
                        <ShieldCheck className="w-3 h-3 mr-1" /> PROTEÇÃO ATIVA
                    </Badge>
                )}
              </div>
            </div>
          </div>
          
          <Button 
            size="lg"
            onClick={handleToggleBot}
            className={`h-14 px-8 text-lg font-bold tracking-wider transition-all duration-300 transform hover:scale-105 shadow-xl border-2
              ${status?.isRunning 
                ? "bg-white text-red-600 hover:bg-red-50 border-transparent" 
                : "bg-transparent text-white border-white hover:bg-white/10"
              }`}
          >
            <Power className={`w-5 h-5 mr-3 ${status?.isRunning ? "animate-pulse" : ""}`} />
            {status?.isRunning ? "DESATIVAR SISTEMA" : "ATIVAR SISTEMA"}
          </Button>
        </div>
      </div>

      {/* Grid de Métricas */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Card Equity */}
        <Card className="glass-panel p-6 relative overflow-hidden group hover:border-red-300 transition-all duration-300">
          <div className="absolute right-0 top-0 w-32 h-32 bg-red-500/5 rounded-full blur-2xl -mr-10 -mt-10 transition-all group-hover:bg-red-500/10"></div>
          <div className="flex flex-col h-full justify-between relative z-10">
            <div>
              <p className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-2">Capital Total</p>
              <div className="flex items-center gap-2">
                <h3 className="text-4xl font-black text-foreground">
                  {status?.equity ? `$${status.equity.toFixed(2)}` : "..."}
                </h3>
                {trend === 'up' && <TrendingUp className="w-6 h-6 text-primary animate-bounce" />}
              </div>
            </div>
            <div className="mt-4 h-16 opacity-50">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={equityData}>
                  <Line type="monotone" dataKey="value" stroke="#ef4444" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Card>

        {/* Card Saldo */}
        <Card className="glass-panel p-6 relative overflow-hidden group hover:border-red-300 transition-all duration-300">
          <div className="absolute right-0 top-0 w-32 h-32 bg-red-500/5 rounded-full blur-2xl -mr-10 -mt-10 transition-all group-hover:bg-red-500/10"></div>
          <div className="flex justify-between items-start relative z-10">
            <div>
              <p className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-2">Disponível</p>
              <h3 className="text-4xl font-black text-primary">
                {status?.balance ? `$${status.balance.toFixed(2)}` : "..."}
              </h3>
            </div>
            <div className="p-3 bg-red-50 rounded-xl">
              <DollarSign className="w-6 h-6 text-primary" />
            </div>
          </div>
          <div className="mt-6 flex items-center gap-2 text-sm text-muted-foreground font-medium">
            <Wallet className="w-4 h-4" />
            <span>Carteira Spot Binance</span>
          </div>
        </Card>

        {/* Card Performance */}
        <Card className="glass-panel p-6 flex items-center justify-between hover:border-red-300 transition-all duration-300">
           <div className="space-y-4">
              <p className="text-sm font-bold text-muted-foreground uppercase tracking-wider">Trades Hoje</p>
              <div className="flex items-baseline gap-2">
                <span className="text-5xl font-black text-foreground">{status?.dailyTrades || 0}</span>
                <span className="text-lg font-bold text-muted-foreground">/ 10</span>
              </div>
              <Badge className="bg-red-100 text-red-700 hover:bg-red-200 border-none px-3 py-1 text-xs font-bold">
                <Target className="w-3 h-3 mr-1" /> META DIÁRIA
              </Badge>
           </div>
           <div className="w-24 h-24 relative">
             <ResponsiveContainer width="100%" height="100%">
               <PieChart>
                 <Pie
                   data={winRateData.length > 0 ? winRateData : [{name: 'Empty', value: 1}]}
                   innerRadius={30}
                   outerRadius={40}
                   dataKey="value"
                   stroke="none"
                 >
                   {winRateData.map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                   ))}
                   {winRateData.length === 0 && <Cell fill="#fee2e2" />}
                 </Pie>
               </PieChart>
             </ResponsiveContainer>
             <div className="absolute inset-0 flex items-center justify-center">
                <Activity className="w-5 h-5 text-primary opacity-50" />
             </div>
           </div>
        </Card>
      </div>

      {/* Gráfico Principal */}
      <Card className="glass-panel p-6 border-red-100/80 shadow-2xl shadow-red-500/10">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-red-50 flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-black text-foreground">CURVA DE CRESCIMENTO</h3>
              <p className="text-xs font-bold text-primary animate-pulse">● ATUALIZAÇÃO EM TEMPO REAL</p>
            </div>
          </div>
        </div>
        
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={equityData}>
              <defs>
                <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.2}/>
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(239, 68, 68, 0.1)" vertical={false} />
              <XAxis 
                dataKey="time" 
                stroke="#9ca3af" 
                fontSize={12} 
                tickLine={false}
                axisLine={false}
                minTickGap={30}
                fontWeight={500}
              />
              <YAxis 
                domain={['auto', 'auto']} 
                stroke="#9ca3af" 
                fontSize={12} 
                tickLine={false}
                axisLine={false}
                tickFormatter={(val) => `$${val}`}
                fontWeight={500}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: "rgba(255, 255, 255, 0.95)", 
                  border: "2px solid #fee2e2", 
                  borderRadius: "1rem", 
                  boxShadow: "0 10px 15px -3px rgba(220, 38, 38, 0.1)",
                  fontWeight: "bold"
                }}
                itemStyle={{ color: "#ef4444" }}
                labelStyle={{ color: "#6b7280", marginBottom: "0.25rem", fontSize: "0.8rem" }}
              />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="#ef4444" 
                strokeWidth={4}
                fill="url(#colorEquity)" 
                animationDuration={1000}
              />
            </AreaChart>
          </ResponsiveContainer>
          
          {equityData.length === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
              <div className="p-4 bg-white/80 backdrop-blur rounded-2xl border border-red-100 shadow-xl text-center">
                <Activity className="w-8 h-8 text-primary animate-spin mb-2 mx-auto" />
                <p className="font-bold text-foreground">Sincronizando...</p>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;