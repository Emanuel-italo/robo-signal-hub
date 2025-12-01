import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ArrowUpRight, ArrowDownRight, History, Trophy, TrendingUp, Activity, Ban } from "lucide-react";
import { api, TradeHistoryItem } from "@/services/api";

const TradeHistory = () => {
  const { data: trades } = useQuery({
    queryKey: ["botHistory"],
    queryFn: api.getHistory,
    refetchInterval: 5000,
  });

  // Cálculo de estatísticas em tempo real para o layout (sem lógica de negócio complexa, apenas visualização)
  const stats = useMemo(() => {
    if (!trades || trades.length === 0) return { total: 0, wins: 0, losses: 0, winRate: 0, totalPnl: 0 };
    
    const total = trades.length;
    const wins = trades.filter(t => t.pnl > 0).length;
    const losses = total - wins;
    const totalPnl = trades.reduce((acc, t) => acc + t.pnl, 0);
    
    return {
      total,
      wins,
      losses,
      winRate: (wins / total) * 100,
      totalPnl
    };
  }, [trades]);

  return (
    <div className="space-y-6 animate-in fade-in duration-700">
      
      {/* Cabeçalho da Seção */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-red-950/30 border border-red-900/50 rounded-xl shadow-[0_0_15px_rgba(220,38,38,0.2)]">
            <History className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h2 className="text-2xl font-black text-white tracking-wide">HISTÓRICO DE EXECUÇÃO</h2>
            <p className="text-gray-500 font-medium text-sm">Registro imutável de todas as operações realizadas</p>
          </div>
        </div>
      </div>

      {/* Cartões de Resumo Rápido (Novidade Visual) */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="glass-panel p-4 border-l-4 border-l-primary flex items-center justify-between relative overflow-hidden">
            <div className="absolute right-0 top-0 opacity-10 -mr-4 -mt-4"><Activity className="w-24 h-24" /></div>
            <div>
                <p className="text-xs text-gray-400 font-bold uppercase">Total de Trades</p>
                <p className="text-2xl font-black text-white">{stats.total}</p>
            </div>
        </Card>

        <Card className="glass-panel p-4 border-l-4 border-l-emerald-500 flex items-center justify-between relative overflow-hidden">
            <div className="absolute right-0 top-0 opacity-10 -mr-4 -mt-4"><Trophy className="w-24 h-24" /></div>
            <div>
                <p className="text-xs text-gray-400 font-bold uppercase">Win Rate</p>
                <div className="flex items-baseline gap-2">
                    <p className="text-2xl font-black text-emerald-400">{stats.winRate.toFixed(1)}%</p>
                    <span className="text-xs text-emerald-500/80 font-bold">({stats.wins} wins)</span>
                </div>
            </div>
        </Card>

        <Card className="glass-panel p-4 border-l-4 border-l-red-500 flex items-center justify-between relative overflow-hidden">
            <div className="absolute right-0 top-0 opacity-10 -mr-4 -mt-4"><Ban className="w-24 h-24" /></div>
            <div>
                <p className="text-xs text-gray-400 font-bold uppercase">Losses</p>
                <p className="text-2xl font-black text-red-400">{stats.losses}</p>
            </div>
        </Card>

        <Card className="glass-panel p-4 border-l-4 border-l-amber-500 flex items-center justify-between relative overflow-hidden">
            <div className="absolute right-0 top-0 opacity-10 -mr-4 -mt-4"><TrendingUp className="w-24 h-24" /></div>
            <div>
                <p className="text-xs text-gray-400 font-bold uppercase">Resultado Total</p>
                <p className={`text-2xl font-black ${stats.totalPnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {stats.totalPnl >= 0 ? "+" : ""}{stats.totalPnl.toFixed(2)}
                </p>
            </div>
        </Card>
      </div>

      {/* Tabela Melhorada */}
      <Card className="glass-panel overflow-hidden border-red-900/30 flex flex-col h-[500px]">
        <div className="overflow-auto custom-scrollbar flex-1">
          <Table>
            <TableHeader className="bg-black/80 backdrop-blur-md sticky top-0 z-10 border-b border-white/10">
              <TableRow className="border-white/5 hover:bg-transparent">
                <TableHead className="text-gray-400 font-bold text-xs uppercase w-[180px]">Data / Hora</TableHead>
                <TableHead className="text-gray-400 font-bold text-xs uppercase">Par de Ativos</TableHead>
                <TableHead className="text-gray-400 font-bold text-xs uppercase text-center">Status</TableHead>
                <TableHead className="text-gray-400 font-bold text-xs uppercase text-right">Preço Ent.</TableHead>
                <TableHead className="text-gray-400 font-bold text-xs uppercase text-right">Preço Saída</TableHead>
                <TableHead className="text-right text-gray-400 font-bold text-xs uppercase pr-6">PnL (Resultado)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trades?.map((trade: TradeHistoryItem, idx: number) => (
                <TableRow key={idx} className="border-white/5 hover:bg-white/5 transition-colors group">
                  <TableCell className="text-gray-300 font-mono text-xs font-medium">
                    {new Date(trade.timestamp).toLocaleString('pt-BR')}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                        <Badge variant="outline" className="bg-white/5 text-white border-white/10 font-bold hover:bg-white/10 transition-colors">
                        {trade.symbol}
                        </Badge>
                    </div>
                  </TableCell>
                  <TableCell className="text-center">
                    <Badge className={`px-3 ${trade.pnl > 0 
                        ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20" 
                        : "bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20"}`}>
                      {trade.pnl > 0 ? "WIN" : "LOSS"}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right text-gray-400 font-mono text-sm">${trade.entry_price?.toFixed(2)}</TableCell>
                  <TableCell className="text-right text-gray-400 font-mono text-sm">${trade.exit_price?.toFixed(2)}</TableCell>
                  <TableCell className="text-right pr-6">
                    <div className="flex items-center justify-end gap-2">
                      <span className={`font-black font-mono text-sm ${trade.pnl > 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {trade.pnl > 0 ? "+" : ""}{trade.pnl?.toFixed(2)}
                      </span>
                      {trade.pnl > 0 ? <ArrowUpRight className="w-4 h-4 text-emerald-500" /> : <ArrowDownRight className="w-4 h-4 text-red-500" />}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
              {(!trades || trades.length === 0) && (
                <TableRow>
                  <TableCell colSpan={6} className="h-64 text-center">
                    <div className="flex flex-col items-center justify-center text-gray-500 opacity-50 gap-4">
                        <History className="w-12 h-12" />
                        <p className="font-medium">Nenhuma operação registrada no banco de dados.</p>
                    </div>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </Card>
    </div>
  );
};

export default TradeHistory;