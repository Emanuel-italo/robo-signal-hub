import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Wallet, DollarSign, Activity, Power, TrendingUp, TrendingDown, Trophy, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from "recharts";
import { api } from "@/services/api";
import { Badge } from "@/components/ui/badge";

// Interfaces
interface BotStatus {
  isRunning: boolean;
  balance: number;
  equity: number;
  dailyTrades: number;
  totalTrades: number;
}

interface ChartData {
  time: string;
  value: number;
}

interface Trade {
  pnl: number;
}

const COLORS = ['#10b981', '#ef4444']; // Verde (Win) e Vermelho (Loss)

const Dashboard = () => {
  // 1. Busca Status do Robô (Equity, Saldo)
  const { data: rawStatus, refetch: refetchStatus } = useQuery({
    queryKey: ["botStatus"],
    queryFn: api.getStatus,
    refetchInterval: 2000, 
  });

  // 2. Busca Histórico para calcular Win Rate
  const { data: history } = useQuery({
    queryKey: ["botHistory"],
    queryFn: api.getHistory,
    refetchInterval: 5000,
  });

  const status = rawStatus as BotStatus | undefined;
  
  // Estado para gráficos
  const [equityData, setEquityData] = useState<ChartData[]>([]);
  const [balanceData, setBalanceData] = useState<ChartData[]>([]); // Para o mini gráfico
  const [winRateData, setWinRateData] = useState<{name: string, value: number}[]>([]);
  const [trend, setTrend] = useState<'up' | 'down' | 'neutral'>('neutral');

  // Atualiza Gráficos de Linha (Equity e Balance)
  useEffect(() => {
    if (status?.equity) {
      const now = new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      
      // Atualiza Equity (Gráfico Principal)
      setEquityData(prev => {
        const newData = [...prev, { time: now, value: status.equity }];
        // Detecta tendência (compara último valor com penúltimo)
        if (prev.length > 0) {
          const last = prev[prev.length - 1].value;
          setTrend(status.equity > last ? 'up' : status.equity < last ? 'down' : 'neutral');
        }
        return newData.slice(-30); // Mantém 30 pontos
      });

      // Atualiza Balance (Mini Gráfico)
      setBalanceData(prev => {
        const newData = [...prev, { time: now, value: status.balance }];
        return newData.slice(-15); // Mantém 15 pontos para o card menor
      });
    }
  }, [status]);

  // Calcula Win Rate para o Gráfico de Pizza
  useEffect(() => {
    if (history && Array.isArray(history)) {
      const wins = history.filter((t: Trade) => t.pnl > 0).length;
      const losses = history.filter((t: Trade) => t.pnl <= 0).length;
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

  // Componente de Mini Gráfico (Sparkline)
  const MiniChart = ({ data, color }: { data: ChartData[], color: string }) => (
    <div className="h-[40px] w-[80px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      
      {/* --- CABEÇALHO --- */}
      <div className="flex flex-col sm:flex-row justify-between items-center gap-4 bg-card/30 p-4 rounded-xl border border-border/50 backdrop-blur-sm">
        <div>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Activity className="w-6 h-6 text-primary" />
            Painel de Controle
          </h2>
          <p className="text-muted-foreground text-sm">Visão geral do desempenho em tempo real.</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden md:flex flex-col items-end mr-4">
            <span className="text-xs text-muted-foreground uppercase font-bold">Status do Sistema</span>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${status?.isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-destructive'}`} />
              <span className={`font-mono font-medium ${status?.isRunning ? 'text-emerald-500' : 'text-destructive'}`}>
                {status?.isRunning ? 'ONLINE' : 'PARADO'}
              </span>
            </div>
          </div>
          <Button 
            size="lg"
            variant={status?.isRunning ? "destructive" : "default"}
            onClick={handleToggleBot}
            className="gap-2 shadow-lg hover:scale-105 transition-transform"
          >
            <Power className="w-4 h-4" />
            {status?.isRunning ? "PARAR ROBÔ" : "INICIAR ROBÔ"}
          </Button>
        </div>
      </div>

      {/* --- GRID DE STATUS (CARDS) --- */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        
        {/* Card Equity (Com Mini Gráfico e Tendência) */}
        <Card className="p-6 bg-card border-border relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Wallet className="w-16 h-16" />
          </div>
          <div className="flex justify-between items-start">
            <div className="space-y-1">
              <p className="text-sm font-medium text-muted-foreground">Capital Total (Equity)</p>
              <div className="flex items-center gap-2">
                <p className="text-3xl font-bold text-foreground">
                  {status?.equity ? `$${status.equity.toFixed(2)}` : "..."}
                </p>
                {trend === 'up' && <TrendingUp className="w-5 h-5 text-emerald-500 animate-bounce" />}
                {trend === 'down' && <TrendingDown className="w-5 h-5 text-red-500 animate-bounce" />}
              </div>
            </div>
            <MiniChart data={equityData} color="#3b82f6" />
          </div>
        </Card>

        {/* Card Saldo (Com Mini Gráfico) */}
        <Card className="p-6 bg-card border-border relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <DollarSign className="w-16 h-16" />
          </div>
          <div className="flex justify-between items-start">
            <div className="space-y-1">
              <p className="text-sm font-medium text-muted-foreground">Saldo Disponível</p>
              <p className="text-3xl font-bold text-emerald-500">
                {status?.balance ? `$${status.balance.toFixed(2)}` : "..."}
              </p>
            </div>
            <MiniChart data={balanceData} color="#10b981" />
          </div>
        </Card>

        {/* Card Trades (Com Win Rate) */}
        <Card className="p-6 bg-card border-border flex items-center justify-between relative overflow-hidden">
           <div className="space-y-2 z-10">
              <p className="text-sm font-medium text-muted-foreground">Performance Hoje</p>
              <div className="flex items-baseline gap-2">
                <p className="text-3xl font-bold text-foreground">{status?.dailyTrades || 0}</p>
                <span className="text-sm text-muted-foreground">/ 5 trades</span>
              </div>
              <div className="flex gap-2 mt-2">
                <Badge variant="outline" className="bg-emerald-500/10 text-emerald-500 border-emerald-500/20 gap-1">
                  <Trophy className="w-3 h-3" /> Wins
                </Badge>
                <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500/20 gap-1">
                  <XCircle className="w-3 h-3" /> Loss
                </Badge>
              </div>
           </div>
           
           {/* Gráfico de Pizza Pequeno */}
           <div className="w-[80px] h-[80px]">
             <ResponsiveContainer width="100%" height="100%">
               <PieChart>
                 <Pie
                   data={winRateData.length > 0 ? winRateData : [{name: 'Empty', value: 1}]}
                   innerRadius={25}
                   outerRadius={35}
                   paddingAngle={2}
                   dataKey="value"
                   stroke="none"
                 >
                   {winRateData.map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                   ))}
                   {winRateData.length === 0 && <Cell fill="#334155" />}
                 </Pie>
               </PieChart>
             </ResponsiveContainer>
           </div>
        </Card>
      </div>

      {/* --- GRÁFICO PRINCIPAL --- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[400px]">
        <Card className="col-span-1 lg:col-span-3 p-6 bg-card border-border flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                Evolução do Capital (Ao Vivo)
              </h3>
              <p className="text-xs text-muted-foreground">Atualização a cada 2 segundos</p>
            </div>
            <Badge variant="secondary" className="font-mono">
              Live Feed
            </Badge>
          </div>
          
          <div className="flex-1 w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={equityData}>
                <defs>
                  <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} vertical={false} />
                <XAxis 
                  dataKey="time" 
                  stroke="hsl(var(--muted-foreground))" 
                  fontSize={12} 
                  tickLine={false}
                  axisLine={false}
                  minTickGap={30}
                />
                <YAxis 
                  domain={['auto', 'auto']} 
                  stroke="hsl(var(--muted-foreground))" 
                  fontSize={12} 
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(val) => `$${val}`}
                  width={60}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: "hsl(var(--popover))", 
                    border: "1px solid hsl(var(--border))", 
                    borderRadius: "0.5rem",
                    boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)"
                  }}
                  itemStyle={{ color: "hsl(var(--foreground))" }}
                  labelStyle={{ color: "hsl(var(--muted-foreground))", marginBottom: "0.25rem" }}
                />
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#3b82f6" 
                  strokeWidth={3}
                  fill="url(#colorEquity)" 
                  isAnimationActive={false} // Desativa animação inicial para fluidez em tempo real
                />
              </AreaChart>
            </ResponsiveContainer>
            
            {equityData.length === 0 && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/50 backdrop-blur-[1px] z-10 rounded-lg">
                <Activity className="w-10 h-10 text-muted-foreground animate-pulse mb-2" />
                <p className="text-muted-foreground font-medium">Aguardando dados do mercado...</p>
                <p className="text-xs text-muted-foreground mt-1">Certifique-se que o backend está rodando na porta 8000</p>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;