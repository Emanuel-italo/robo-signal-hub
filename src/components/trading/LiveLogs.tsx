import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Terminal, Cpu, Radio, ShieldCheck } from "lucide-react";
import { api } from "@/services/api";

const LiveLogs = () => {
  const { data: logs } = useQuery({
    queryKey: ["botLogs"],
    queryFn: api.getLogs,
    refetchInterval: 1000,
  });

  const getLogStyle = (level: string) => {
    switch(level) {
      case "INFO": return "text-blue-400 border-blue-400/20 bg-blue-400/5";
      case "WARNING": return "text-yellow-500 border-yellow-500/20 bg-yellow-500/5";
      case "ERROR": return "text-red-500 border-red-500/20 bg-red-500/10";
      case "SUCCESS": return "text-emerald-500 border-emerald-500/20 bg-emerald-500/5";
      default: return "text-gray-400 border-gray-400/20";
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      
      {/* Status do Terminal */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="glass-panel p-4 flex items-center gap-3">
          <div className="p-2 bg-emerald-500/10 rounded-lg"><Cpu className="w-5 h-5 text-emerald-500" /></div>
          <div><p className="text-xs text-gray-500 font-bold">CPU CORE</p><p className="text-white font-mono">ATIVO</p></div>
        </Card>
        <Card className="glass-panel p-4 flex items-center gap-3">
          <div className="p-2 bg-blue-500/10 rounded-lg"><Radio className="w-5 h-5 text-blue-500 animate-pulse" /></div>
          <div><p className="text-xs text-gray-500 font-bold">LATÊNCIA</p><p className="text-white font-mono">24ms</p></div>
        </Card>
        <Card className="glass-panel p-4 flex items-center gap-3">
          <div className="p-2 bg-red-500/10 rounded-lg"><ShieldCheck className="w-5 h-5 text-primary" /></div>
          <div><p className="text-xs text-gray-500 font-bold">SEGURANÇA</p><p className="text-white font-mono">BLINDADO</p></div>
        </Card>
      </div>

      {/* Console */}
      <Card className="glass-panel border-red-900/30 overflow-hidden flex flex-col h-[600px] shadow-2xl shadow-black">
        <div className="p-3 border-b border-white/5 bg-black/40 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-primary" />
            <span className="text-xs font-bold text-gray-400 tracking-widest">SYSTEM_KERNEL_V1</span>
          </div>
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500/20" />
            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20" />
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/20" />
          </div>
        </div>
        
        <div className="flex-1 p-4 overflow-y-auto space-y-2 font-mono text-xs bg-black/80">
          {logs?.map((log: any, idx: number) => (
            <div key={idx} className="flex gap-3 hover:bg-white/5 p-1 rounded transition-colors group">
              <span className="text-gray-600 select-none group-hover:text-gray-400">[{log.time}]</span>
              <Badge variant="outline" className={`h-5 px-1.5 rounded-sm border ${getLogStyle(log.level)}`}>
                {log.level}
              </Badge>
              <span className={`flex-1 break-all ${log.level === 'ERROR' ? 'text-red-400' : 'text-gray-300'}`}>
                {log.message}
              </span>
            </div>
          ))}
          {!logs && (
            <div className="text-center py-20">
              <span className="inline-block w-2 h-4 bg-primary animate-pulse"/>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default LiveLogs;