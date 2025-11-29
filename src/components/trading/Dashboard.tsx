import { Card } from "@/components/ui/card";
import { ArrowUpRight, ArrowDownRight, Wallet, TrendingUp, Activity, DollarSign } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts";

const mockEquityData = [
  { time: "00:00", value: 10000 },
  { time: "04:00", value: 10150 },
  { time: "08:00", value: 10080 },
  { time: "12:00", value: 10320 },
  { time: "16:00", value: 10280 },
  { time: "20:00", value: 10450 },
  { time: "24:00", value: 10580 },
];

const mockPerformanceData = [
  { symbol: "BTC/USDT", trades: 15, winRate: 73.3, pnl: 1245.80 },
  { symbol: "ETH/USDT", trades: 12, winRate: 66.7, pnl: 892.50 },
  { symbol: "SOL/USDT", trades: 8, winRate: 62.5, pnl: 456.20 },
];

const Dashboard = () => {
  const stats = [
    {
      label: "Saldo Total",
      value: "$10,580.00",
      change: "+5.8%",
      trend: "up",
      icon: Wallet,
      color: "text-accent"
    },
    {
      label: "Lucro/Prejuízo",
      value: "+$580.00",
      change: "+5.8%",
      trend: "up",
      icon: DollarSign,
      color: "text-success"
    },
    {
      label: "Trades Hoje",
      value: "8",
      change: "de 5 max",
      trend: "neutral",
      icon: Activity,
      color: "text-warning"
    },
    {
      label: "Win Rate",
      value: "68.2%",
      change: "+2.1%",
      trend: "up",
      icon: TrendingUp,
      color: "text-success"
    },
  ];

  return (
    <div className="space-y-6">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, idx) => (
          <Card key={idx} className="p-6 bg-card border-border hover:border-primary/50 transition-all duration-300">
            <div className="flex items-start justify-between">
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">{stat.label}</p>
                <p className="text-2xl font-bold text-foreground">{stat.value}</p>
                <div className="flex items-center gap-1">
                  {stat.trend === "up" && <ArrowUpRight className="w-4 h-4 text-success" />}
                  {stat.trend === "down" && <ArrowDownRight className="w-4 h-4 text-destructive" />}
                  <span className={`text-sm ${stat.trend === "up" ? "text-success" : stat.trend === "down" ? "text-destructive" : "text-muted-foreground"}`}>
                    {stat.change}
                  </span>
                </div>
              </div>
              <div className={`w-12 h-12 rounded-lg bg-secondary/50 flex items-center justify-center ${stat.color}`}>
                <stat.icon className="w-6 h-6" />
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Equity Curve */}
        <Card className="p-6 bg-card border-border">
          <h3 className="text-lg font-semibold text-foreground mb-4">Curva de Capital</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={mockEquityData}>
              <defs>
                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" />
              <YAxis stroke="hsl(var(--muted-foreground))" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: "hsl(var(--card))", 
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "0.5rem"
                }}
              />
              <Area type="monotone" dataKey="value" stroke="hsl(var(--primary))" fill="url(#colorValue)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        {/* Performance by Symbol */}
        <Card className="p-6 bg-card border-border">
          <h3 className="text-lg font-semibold text-foreground mb-4">Performance por Símbolo</h3>
          <div className="space-y-4">
            {mockPerformanceData.map((item, idx) => (
              <div key={idx} className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg hover:bg-secondary/50 transition-colors">
                <div className="flex-1">
                  <p className="font-medium text-foreground">{item.symbol}</p>
                  <p className="text-sm text-muted-foreground">{item.trades} trades</p>
                </div>
                <div className="text-right mr-6">
                  <p className="text-sm text-muted-foreground">Win Rate</p>
                  <p className="font-semibold text-success">{item.winRate}%</p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-muted-foreground">P&L</p>
                  <p className={`font-semibold ${item.pnl > 0 ? "text-success" : "text-destructive"}`}>
                    ${item.pnl.toFixed(2)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Market Sentiment */}
      <Card className="p-6 bg-card border-border">
        <h3 className="text-lg font-semibold text-foreground mb-4">Análise de Sentimento do Mercado</h3>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <div className="flex justify-between mb-2">
              <span className="text-sm text-muted-foreground">Sentimento</span>
              <span className="text-sm font-medium text-success">Positivo (0.342)</span>
            </div>
            <div className="w-full h-3 bg-secondary rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-destructive via-warning to-success rounded-full" style={{ width: "67%" }}></div>
            </div>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-success">+67%</p>
            <p className="text-sm text-muted-foreground">Bullish</p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;
