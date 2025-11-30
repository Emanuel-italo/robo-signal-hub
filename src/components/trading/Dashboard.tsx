import { useState, useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Wallet, DollarSign, Activity, Power, TrendingUp, Target, BrainCircuit, ShieldCheck, Filter, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from "recharts";
// CORREÇÃO: Agora importamos BotStatus da api, pois o arquivo api.ts exporta ele.
import { api, BotStatus } from "@/services/api"; 
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { useWakeLock } from "@/hooks/use-wake-lock";
import MarketTicker from "./MarketTicker";

interface ChartData {
  time: string;
  value: number;
}

const COLORS = ['#ef4444', '#fca5a5'];

const Dashboard = () => {
  // Busca status com tipagem automática ou inferida
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

  // Cast simples e seguro
  const status = rawStatus as BotStatus | undefined;
  
  const { requestLock, releaseLock } = useWakeLock();
  
  const [equityData, setEquityData] = useState<ChartData[]>([]);
  const [winRateData, setWinRateData] = useState<{name: string, value: number}[]>([]);
  const [trend, setTrend] = useState<'up' | 'down' | 'neutral'>('neutral');
  const [selectedAsset, setSelectedAsset] = useState<string>("GLOBAL");

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

  // Wake Lock
  useEffect(() => {
    if (status?.isRunning) requestLock();
    else releaseLock();
  }, [status?.isRunning, requestLock, releaseLock]);

  // Atualiza Gráfico
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

  // Win Rate
  useEffect(() => {
    if (history && Array.isArray(history)) {
      const wins = history.filter((t: any) => t.pnl > 0).length;
      const losses = history.filter((t: any) => t.pnl <= 0).length;
      
      if (wins + losses > 0) {
        setWinRateData([
          { name: 'Wins', value: wins },
          { name: 'Losses', value: losses }
        ]);
      } else {
        setWinRateData([]);
      }
    }
  }, [history]);

  const handleToggleBot = async () => {
    try {
      if (status?.isRunning) {
        await api.stopBot();
      } else {
        await api.startBot();
      }
      setTimeout(() => refetchStatus(), 500);
    } catch (error) {
      console.error("Erro toggle:", error);
      toast.error("Erro de conexão com o robô");
    }
  };

  // Métricas
  const metrics = useMemo(() => {
    if (!status) return null;

    const positions = status.openPositions || [];

    if (selectedAsset === "GLOBAL") {
        const totalInvested = positions.reduce((acc, pos) => acc + pos.invested, 0);
        const totalPnL = positions.reduce((acc, pos) => acc + pos.pnl, 0);
        
        return {
            title: "PATRIMÔNIO LÍQUIDO",
            value: status.equity,
            subValue: status.balance,
            subLabel: "Disponível (Caixa)",
            pnl: totalPnL,
            pnlPercent: totalInvested > 0 ? (totalPnL / totalInvested) * 100 : 0,
            isGlobal: true
        };
    } else {
        const pos = positions.find(p => p.symbol === selectedAsset);
        if (!pos) return null;

        return {
            title: `POSIÇÃO: ${pos.symbol}`,
            value: pos.invested + pos.pnl,
            subValue: pos.invested,
            subLabel: "Investido Inicialmente",
            pnl: pos.pnl,
            pnlPercent: (pos.pnl / pos.invested) * 100,
            isGlobal: false
        };
    }
  }, [status, selectedAsset]);

  // Reset do filtro se o ativo sumir
  useEffect(() => {
      const positions = status?.openPositions || [];
      if (status && selectedAsset !== "GLOBAL" && !positions.find(p => p.symbol === selectedAsset)) {
          setSelectedAsset("GLOBAL");
      }
  }, [status, selectedAsset]);

  return (
    <div className="space-y-8 animate-in fade-in zoom-in-95 duration-700">

      {/* Ticker - Agora tipado corretamente */}
      {status && status.marketPrices && (
        <div className="rounded-xl border border-red-900/30 overflow-hidden shadow-lg bg-black/40 backdrop-blur-sm">
           <MarketTicker prices={status.marketPrices} />
        </div>
      )}
      
      {/* Banner */}
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

      {/* Filtros */}
      <div className="flex flex-col md:flex-row gap-4 items-stretch md:items-center justify-between bg-black/40 p-4 rounded-xl border border-white/5 backdrop-blur-md">
        
        <div className="flex items-center gap-4 flex-1">
            <div className="p-2 bg-red-500/10 rounded-lg border border-red-500/20">
                <Filter className="w-5 h-5 text-primary" />
            </div>
            <div className="flex-1">
                <p className="text-[10px] text-gray-500 font-bold uppercase mb-1">Visualização do Painel</p>
                <Select value={selectedAsset} onValueChange={setSelectedAsset}>
                    <SelectTrigger className="w-full md:w-[250px] bg-black/50 border-red-900/30 text-white font-bold h-10">
                        <SelectValue placeholder="Selecione..." />
                    </SelectTrigger>
                    <SelectContent className="bg-black/95 border-red-900 text-white">
                        <SelectItem value="GLOBAL" className="font-bold text-red-400">⚡ PORTFÓLIO GLOBAL</SelectItem>
                        {status?.openPositions?.map((pos) => (
                            <SelectItem key={pos.symbol} value={pos.symbol} className="font-mono">
                                {pos.symbol}
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
            </div>
        </div>

        {metrics && (
            <div className={`flex items-center gap-4 px-6 py-2 rounded-xl border ${metrics.pnl >= 0 ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-red-500/10 border-red-500/30'}`}>
                <div className="text-right">
                    <p className={`text-[10px] font-bold uppercase ${metrics.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {metrics.isGlobal ? 'Resultado Aberto (PnL)' : 'Resultado da Operação'}
                    </p>
                    <div className="flex items-center justify-end gap-2">
                        {metrics.pnl >= 0 ? <ArrowUpRight className="w-5 h-5 text-emerald-500" /> : <ArrowDownRight className="w-5 h-5 text-red-500" />}
                        <span className={`text-2xl font-black font-mono tracking-tighter ${metrics.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {metrics.pnl >= 0 ? "+" : ""}{metrics.pnl.toFixed(2)}
                        </span>
                        <Badge variant="outline" className={`ml-2 font-bold ${metrics.pnl >= 0 ? 'border-emerald-500 text-emerald-400' : 'border-red-500 text-red-400'}`}>
                            {metrics.pnlPercent.toFixed(2)}%
                        </Badge>
                    </div>
                </div>
            </div>
        )}
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        <Card className="glass-panel p-6 relative overflow-hidden group hover:border-red-300 transition-all duration-300">
          <div className="absolute right-0 top-0 w-32 h-32 bg-red-500/5 rounded-full blur-2xl -mr-10 -mt-10 transition-all group-hover:bg-red-500/10"></div>
          <div className="flex flex-col h-full justify-between relative z-10">
            <div>
              <p className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-2">{metrics?.title || "Carregando..."}</p>
              <div className="flex items-center gap-2">
                <h3 className="text-4xl font-black text-foreground">
                  {metrics ? `$${metrics.value.toFixed(2)}` : "..."}
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

        <Card className="glass-panel p-6 relative overflow-hidden group hover:border-red-300 transition-all duration-300">
          <div className="absolute right-0 top-0 w-32 h-32 bg-red-500/5 rounded-full blur-2xl -mr-10 -mt-10 transition-all group-hover:bg-red-500/10"></div>
          <div className="flex justify-between items-start relative z-10">
            <div>
              <p className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-2">{metrics?.subLabel || "..."}</p>
              <h3 className="text-4xl font-black text-primary">
                {metrics ? `$${metrics.subValue.toFixed(2)}` : "..."}
              </h3>
            </div>
            <div className="p-3 bg-red-50 rounded-xl">
              <DollarSign className="w-6 h-6 text-primary" />
            </div>
          </div>
          <div className="mt-6 flex items-center gap-2 text-sm text-muted-foreground font-medium">
            <Wallet className="w-4 h-4" />
            <span>{selectedAsset === "GLOBAL" ? "Carteira Spot Binance" : "Alocado nesta ordem"}</span>
          </div>
        </Card>

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
              <h3 className="text-lg font-black text-foreground">
                {selectedAsset === "GLOBAL" ? "CURVA DE PATRIMÔNIO GLOBAL" : `DESEMPENHO DO PORTFÓLIO (FOCANDO EM ${selectedAsset})`}
              </h3>
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