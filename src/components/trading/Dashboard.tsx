import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Wallet, DollarSign, Activity, Power, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from "recharts";
import { api } from "@/services/api";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

// Tipagem
interface BotStatus {
  isRunning: boolean;
  balance: number;
  equity: number;
  dailyTrades: number;
}

interface ChartData {
  time: string;
  value: number;
}

const Dashboard = () => {
  const { data: rawStatus, refetch } = useQuery({
    queryKey: ["botStatus"],
    queryFn: api.getStatus,
    refetchInterval: 2000, 
  });

  const status = rawStatus as BotStatus | undefined;
  const [chartData, setChartData] = useState<ChartData[]>([]);

  useEffect(() => {
    if (status?.equity) {
      setChartData(prev => {
        const newData = [...prev, { 
            time: new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }), 
            value: status.equity 
        }];
        return newData.slice(-30); // Mostra os últimos 30 pontos
      });
    }
  }, [status]);

  const handleToggleBot = async () => {
    if (status?.isRunning) await api.stopBot();
    else await api.startBot();
    refetch();
  };

  // Configuração dos Cards com Explicações (Tooltips)
  const stats = [
    {
      label: "Capital Total (Equity)",
      value: status?.equity ? `$${status.equity.toFixed(2)}` : "...",
      icon: Wallet,
      color: "text-accent",
      help: "Soma do seu saldo livre + valor atual investido nas posições abertas."
    },
    {
      label: "Saldo Disponível",
      value: status?.balance ? `$${status.balance.toFixed(2)}` : "...",
      icon: DollarSign,
      color: "text-success",
      help: "Dinheiro livre na carteira pronto para ser usado em novas compras."
    },
    {
      label: "Trades Hoje",
      value: status ? `${status.dailyTrades} / 5` : "...",
      icon: Activity,
      color: "text-warning",
      help: "Número de operações realizadas hoje vs Limite diário de segurança."
    },
  ];

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Painel de Controle</h2>
          <p className="text-muted-foreground">Monitoramento em tempo real do robô.</p>
        </div>
        <Button 
          size="lg"
          variant={status?.isRunning ? "destructive" : "default"}
          onClick={handleToggleBot}
          className="gap-2 w-full sm:w-auto shadow-lg transition-all hover:scale-105"
        >
          <Power className="w-4 h-4" />
          {status?.isRunning ? "PARAR ROBÔ" : "INICIAR ROBÔ"}
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {stats.map((stat, idx) => (
          <Card key={idx} className="p-6 bg-card border-border hover:border-primary/50 transition-colors group relative overflow-hidden">
            {/* Efeito de brilho no fundo */}
            <div className={`absolute -right-6 -top-6 w-24 h-24 bg-current opacity-5 rounded-full blur-2xl ${stat.color}`} />
            
            <div className="flex items-start justify-between relative z-10">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
                  
                  {/* Tooltip de Ajuda */}
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="w-3.5 h-3.5 text-muted-foreground/50 hover:text-primary transition-colors cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs">{stat.help}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <p className="text-3xl font-bold text-foreground">{stat.value}</p>
              </div>
              <div className={`w-12 h-12 rounded-xl bg-secondary/50 flex items-center justify-center ${stat.color} group-hover:scale-110 transition-transform`}>
                <stat.icon className="w-6 h-6" />
              </div>
            </div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6">
        <Card className="p-6 bg-card border-border">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-foreground">Curva de Capital (Ao Vivo)</h3>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"/>
              Atualizando
            </div>
          </div>
          <div className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
                <XAxis 
                  dataKey="time" 
                  stroke="hsl(var(--muted-foreground))" 
                  fontSize={12} 
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis 
                  domain={['auto', 'auto']} 
                  stroke="hsl(var(--muted-foreground))" 
                  fontSize={12} 
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => `$${value}`}
                />
                <RechartsTooltip 
                  contentStyle={{ 
                    backgroundColor: "hsl(var(--card))", 
                    border: "1px solid hsl(var(--border))", 
                    borderRadius: "0.5rem",
                    color: "hsl(var(--foreground))"
                  }}
                />
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#10b981" 
                  fill="url(#colorValue)" 
                  strokeWidth={3} 
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
            {chartData.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center bg-background/50 backdrop-blur-[1px]">
                <p className="text-muted-foreground animate-pulse">Aguardando dados do robô...</p>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;