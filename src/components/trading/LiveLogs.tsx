// src/components/trading/LiveLogs.tsx
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Terminal } from "lucide-react";
import { api } from "@/services/api";

const LiveLogs = () => {
  const { data: logs } = useQuery({
    queryKey: ["botLogs"],
    queryFn: api.getLogs,
    refetchInterval: 1000, // Atualiza a cada 1 segundo
  });

  const getLevelColor = (level: string) => {
    switch(level) {
      case "SUCCESS": return "text-success border-success/30";
      case "WARNING": return "text-warning border-warning/30";
      case "ERROR": return "text-destructive border-destructive/30";
      default: return "text-accent border-accent/30";
    }
  };

  return (
    <Card className="bg-card border-border h-[600px] flex flex-col">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Terminal className="w-5 h-5" />
          Logs do Python
        </h3>
      </div>
      <div className="p-4 space-y-2 overflow-y-auto flex-1 font-mono text-xs">
        {logs?.map((log: any, idx: number) => (
          <div key={idx} className="flex gap-2">
            <span className="text-muted-foreground">[{log.time}]</span>
            <Badge variant="outline" className={`${getLevelColor(log.level)} h-5 px-1`}>
              {log.level}
            </Badge>
            <span className="text-foreground">{log.message}</span>
          </div>
        ))}
        {!logs && <p className="text-muted-foreground">Conectando ao Python...</p>}
      </div>
    </Card>
  );
};

export default LiveLogs;