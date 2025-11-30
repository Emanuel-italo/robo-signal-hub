import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Terminal, Cpu, Radio, ShieldCheck, ChevronRight } from "lucide-react";
import { api } from "@/services/api";

const LiveLogs = () => {
  // Use um intervalo menor para parecer mais "live" (500ms)
  const { data: logs } = useQuery({
    queryKey: ["botLogs"],
    queryFn: api.getLogs,
    refetchInterval: 500, 
  });

  // Referência para o final da lista de logs
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Efeito de auto-scroll
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  const getLogStyle = (level: string) => {
    switch(level) {
      case "INFO": return "text-blue-400 border-blue-400/20 bg-blue-400/5 shadow-[0_0_10px_rgba(96,165,250,0.1)]";
      case "WARNING": return "text-yellow-500 border-yellow-500/20 bg-yellow-500/5 shadow-[0_0_10px_rgba(234,179,8,0.1)]";
      case "ERROR": return "text-red-500 border-red-500/20 bg-red-500/10 shadow-[0_0_15px_rgba(239,68,68,0.2)]";
      case "SUCCESS": return "text-emerald-500 border-emerald-500/20 bg-emerald-500/5 shadow-[0_0_10px_rgba(16,185,129,0.1)]";
      default: return "text-gray-400 border-gray-400/20";
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500 h-full">
      
      {/* Status do Terminal */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="glass-panel p-4 flex items-center gap-3 hover:bg-white/5 transition-colors">
          <div className="p-2 bg-emerald-500/10 rounded-lg border border-emerald-500/20"><Cpu className="w-5 h-5 text-emerald-500" /></div>
          <div><p className="text-[10px] text-gray-500 font-bold uppercase">Processamento</p><p className="text-white font-mono text-sm font-bold">ATIVO</p></div>
        </Card>
        <Card className="glass-panel p-4 flex items-center gap-3 hover:bg-white/5 transition-colors">
          <div className="p-2 bg-blue-500/10 rounded-lg border border-blue-500/20"><Radio className="w-5 h-5 text-blue-500 animate-pulse" /></div>
          <div><p className="text-[10px] text-gray-500 font-bold uppercase">Latência da Rede</p><p className="text-white font-mono text-sm font-bold">24ms</p></div>
        </Card>
        <Card className="glass-panel p-4 flex items-center gap-3 hover:bg-white/5 transition-colors">
          <div className="p-2 bg-red-500/10 rounded-lg border border-red-500/20"><ShieldCheck className="w-5 h-5 text-primary" /></div>
          <div><p className="text-[10px] text-gray-500 font-bold uppercase">Status do Sistema</p><p className="text-white font-mono text-sm font-bold">BLINDADO</p></div>
        </Card>
      </div>

      {/* Console Principal */}
      <Card className="glass-panel border-red-900/30 overflow-hidden flex flex-col h-[600px] shadow-2xl shadow-black relative bg-black">
        {/* Header do Console */}
        <div className="p-3 border-b border-white/10 bg-black/60 flex items-center justify-between backdrop-blur-md sticky top-0 z-10">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-primary" />
            <span className="text-xs font-bold text-gray-400 tracking-widest font-mono">SYSTEM_KERNEL_V1.0</span>
          </div>
          <div className="flex items-center gap-2">
             <div className="flex gap-1.5 mr-2">
                <div className="w-2.5 h-2.5 rounded-full bg-red-500/20 border border-red-500/50" />
                <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20 border border-yellow-500/50" />
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/20 border border-emerald-500/50" />
             </div>
          </div>
        </div>
        
        {/* Corpo do Log com Scanlines (efeito retro) */}
        <div className="flex-1 relative overflow-hidden">
            {/* Efeito de scanline CSS */}
            <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-10 bg-[length:100%_4px,3px_100%] opacity-20" />
            
            <div className="h-full overflow-y-auto p-4 space-y-1 font-mono text-xs scrollbar-thin scrollbar-thumb-red-900/20 scrollbar-track-black">
                {logs?.map((log: any, idx: number) => (
                    <div key={idx} className="flex gap-2 hover:bg-white/5 p-1 rounded transition-all duration-200 items-start animate-in slide-in-from-left-2 fade-in">
                        <span className="text-gray-600 select-none min-w-[80px]">[{log.time}]</span>
                        <ChevronRight className="w-3 h-3 text-gray-700 mt-0.5 shrink-0" />
                        <div className="flex-1 flex flex-wrap items-center gap-2">
                            <Badge variant="outline" className={`h-4 px-1 rounded-[2px] border text-[10px] uppercase ${getLogStyle(log.level)}`}>
                                {log.level}
                            </Badge>
                            <span className={`break-all ${log.level === 'ERROR' ? 'text-red-400 font-bold' : 'text-gray-300'}`}>
                                {log.message}
                            </span>
                        </div>
                    </div>
                ))}
                
                {(!logs || logs.length === 0) && (
                    <div className="text-center py-20 flex flex-col items-center gap-2 opacity-50">
                        <span className="inline-block w-2 h-4 bg-primary animate-pulse"/>
                        <span className="text-gray-500">Aguardando conexão com o Neural Core...</span>
                    </div>
                )}
                
                {/* Elemento invisível para forçar o scroll */}
                <div ref={logsEndRef} />
            </div>
        </div>
      </Card>
    </div>
  );
};

export default LiveLogs;