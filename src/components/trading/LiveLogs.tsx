import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Terminal, AlertCircle, CheckCircle, Info, TrendingUp } from "lucide-react";

const mockLogs = [
  { time: "15:23:45", level: "INFO", message: "[SCAN] BTC/USDT prob 0.742 score 0.681 VWAP:ACIMA" },
  { time: "15:23:47", level: "SUCCESS", message: "[BUY SIM] BTC/USDT @ 43890.25 invest: $211.60 score: 0.68" },
  { time: "15:24:12", level: "INFO", message: "[SCAN] ETH/USDT prob 0.658 score 0.612 VWAP:ACIMA" },
  { time: "15:24:15", level: "SUCCESS", message: "[BUY SIM] ETH/USDT @ 2489.50 invest: $199.60 score: 0.61" },
  { time: "15:25:30", level: "INFO", message: "[STATUS] News raw: 0.342 | Saldo: $7605.51" },
  { time: "15:25:31", level: "INFO", message: "Posições abertas:" },
  { time: "15:25:32", level: "INFO", message: "  BTC/USDT: lucro flutuante $14.89" },
  { time: "15:25:33", level: "INFO", message: "  ETH/USDT: lucro flutuante $27.35" },
  { time: "15:26:45", level: "WARNING", message: "[IGNORADO VWAP] SOL/USDT: sinal ok mas abaixo VWAP (contra-tendência)" },
  { time: "15:28:15", level: "INFO", message: "[SCAN] BNB/USDT prob 0.523 score 0.489 VWAP:ABAIXO" },
  { time: "15:30:22", level: "SUCCESS", message: "[FECHOU] BTC/USDT TAKE PROFIT 💰 PnL: $13.18" },
  { time: "15:31:00", level: "INFO", message: "Ciclo completo. Dormindo..." },
];

const LiveLogs = () => {
  const getLevelIcon = (level: string) => {
    switch(level) {
      case "SUCCESS": return <CheckCircle className="w-4 h-4 text-success" />;
      case "WARNING": return <AlertCircle className="w-4 h-4 text-warning" />;
      case "ERROR": return <AlertCircle className="w-4 h-4 text-destructive" />;
      default: return <Info className="w-4 h-4 text-accent" />;
    }
  };

  const getLevelColor = (level: string) => {
    switch(level) {
      case "SUCCESS": return "bg-success/10 text-success border-success/30";
      case "WARNING": return "bg-warning/10 text-warning border-warning/30";
      case "ERROR": return "bg-destructive/10 text-destructive border-destructive/30";
      default: return "bg-accent/10 text-accent border-accent/30";
    }
  };

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-6 bg-card border-border">
          <div className="flex items-center gap-3">
            <Terminal className="w-8 h-8 text-accent" />
            <div>
              <p className="text-sm text-muted-foreground">Sistema</p>
              <p className="text-lg font-bold text-success">Online</p>
            </div>
          </div>
        </Card>
        <Card className="p-6 bg-card border-border">
          <div className="flex items-center gap-3">
            <TrendingUp className="w-8 h-8 text-primary" />
            <div>
              <p className="text-sm text-muted-foreground">Último Scan</p>
              <p className="text-lg font-bold text-foreground">15:28:15</p>
            </div>
          </div>
        </Card>
        <Card className="p-6 bg-card border-border">
          <div className="flex items-center gap-3">
            <CheckCircle className="w-8 h-8 text-success" />
            <div>
              <p className="text-sm text-muted-foreground">Eventos Hoje</p>
              <p className="text-lg font-bold text-foreground">156</p>
            </div>
          </div>
        </Card>
        <Card className="p-6 bg-card border-border">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-8 h-8 text-warning" />
            <div>
              <p className="text-sm text-muted-foreground">Avisos</p>
              <p className="text-lg font-bold text-warning">3</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Logs */}
      <Card className="bg-card border-border">
        <div className="p-4 border-b border-border flex items-center justify-between">
          <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <Terminal className="w-5 h-5" />
            Logs em Tempo Real
          </h3>
          <Badge variant="outline" className="bg-success/10 text-success border-success/30">
            <span className="w-2 h-2 bg-success rounded-full animate-pulse mr-2"></span>
            Live
          </Badge>
        </div>
        <div className="p-4 space-y-2 max-h-[600px] overflow-y-auto font-mono text-sm">
          {mockLogs.map((log, idx) => (
            <div key={idx} className="flex items-start gap-3 p-3 bg-secondary/30 rounded-lg hover:bg-secondary/50 transition-colors">
              <span className="text-muted-foreground whitespace-nowrap">{log.time}</span>
              {getLevelIcon(log.level)}
              <Badge variant="outline" className={`${getLevelColor(log.level)} whitespace-nowrap`}>
                {log.level}
              </Badge>
              <span className="text-foreground flex-1">{log.message}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default LiveLogs;
