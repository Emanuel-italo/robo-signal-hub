import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Wallet, DollarSign, Activity, Power } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { api } from "@/services/api";

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
        const newData = [
          ...prev, 
          { 
            time: new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }), 
            value: status.equity 
          }
        ];
        return newData.slice(-20);
      });
    }
  }, [status]);

  const handleToggleBot = async () => {
    if (status?.isRunning) {
      await api.stopBot();
    } else {
      await api.startBot();
    }
    refetch();
  };

  const stats = [
    {
      label: "Capital Total (Equity)",
      value: status?.equity ? `$${status.equity.toFixed(2)}` : "...",
      icon: Wallet,
      color: "text-accent"
    },
    {
      label: "Saldo Disponível",
      value: status?.balance ? `$${status.balance.toFixed(2)}` : "...",
      icon: DollarSign,
      color: "text-success"
    },
    {
      label: "Trades Hoje",
      value: status ? status.dailyTrades : "...",
      icon: Activity,
      color: "text-warning"
    },
  ];

  return (
    <div className="space-y-6">
      {/* --- CABEÇALHO COM O BOTÃO --- */}
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">Painel de Controle</h2>
        <Button 
          variant={status?.isRunning ? "destructive" : "default"}
          onClick={handleToggleBot}
          className="gap-2 shadow-lg"
        >
          <Power className="w-4 h-4" />
          {status?.isRunning ? "PARAR ROBÔ" : "INICIAR ROBÔ"}
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {stats.map((stat, idx) => (
          <Card key={idx} className="p-6 bg-card border-border">
            <div className="flex items-start justify-between">
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">{stat.label}</p>
                <p className="text-2xl font-bold text-foreground">{stat.value}</p>
              </div>
              <div className={`w-12 h-12 rounded-lg bg-secondary/50 flex items-center justify-center ${stat.color}`}>
                <stat.icon className="w-6 h-6" />
              </div>
            </div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6">
        <Card className="p-6 bg-card border-border">
          <h3 className="text-lg font-semibold text-foreground mb-4">Evolução do Capital (Ao Vivo)</h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                <YAxis domain={['auto', 'auto']} stroke="hsl(var(--muted-foreground))" fontSize={12} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "0.5rem" }}
                />
                <Area type="monotone" dataKey="value" stroke="#10b981" fill="url(#colorValue)" strokeWidth={2} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
            {chartData.length === 0 && (
              <p className="text-center text-muted-foreground text-sm mt-[-150px]">
                Aguardando dados... (Verifique se o backend está rodando)
              </p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;