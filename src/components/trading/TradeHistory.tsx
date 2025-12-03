import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ArrowUpRight, ArrowDownRight, History, Trophy, TrendingUp, Activity, Ban, DollarSign } from "lucide-react";
import { api, TradeHistoryItem } from "@/services/api";

const TradeHistory = () => {
  const { data: trades } = useQuery({
    queryKey: ["botHistory"],
    queryFn: api.getHistory,
    refetchInterval: 5000,
  });

  // Cálculo de estatísticas financeiras detalhadas
  const stats = useMemo(() => {
    if (!trades || !Array.isArray(trades) || trades.length === 0) {
      return { total: 0, wins: 0, losses: 0, winRate: 0, totalPnl: 0, grossProfit: 0, grossLoss: 0, totalInvested: 0 };
    }
    
    const total = trades.length;
    const wins = trades.filter((t: TradeHistoryItem) => t.pnl > 0).length;
    const losses = total - wins;
    const totalPnl = trades.reduce((acc: number, t: TradeHistoryItem) => acc + t.pnl, 0);
    
    // Novos cálculos financeiros absolutos
    const grossProfit = trades
      .filter((t: TradeHistoryItem) => t.pnl > 0)
      .reduce((acc: number, t: TradeHistoryItem) => acc + t.pnl, 0);

    const grossLoss = trades
      .filter((t: TradeHistoryItem) => t.pnl < 0)
      .reduce((acc: number, t: TradeHistoryItem) => acc + t.pnl, 0); // Será negativo

    // Estimativa do volume investido total (baseado em entry_price * quantity se disponível, ou aproximado)
    // Nota: O backend precisa enviar 'quantity' no histórico para ser exato. 
    // Se não tiver, o cálculo individual na tabela pode ficar pendente, mas vamos tentar usar o que tem.
    
    return {
      total,
      wins,
      losses,
      winRate: total > 0 ? (wins / total) * 100 : 0,
      totalPnl,
      grossProfit,
      grossLoss
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
            <h2 className="text-2xl font-black text-white tracking-wide">HISTÓRICO FINANCEIRO</h2>
            <p className="text-gray-500 font-medium text-sm">Detalhamento financeiro de todas as operações encerradas</p>
          </div>
        </div>
      </div>

      {/* Cartões Financeiros (Valores Absolutos) */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        
        {/* Total Ganho (R$) */}
        <Card className="glass-panel p-4 border-l-4 border-l-emerald-500 flex items-center justify-between relative overflow-hidden group hover:bg-emerald-500/5 transition-colors">
            <div className="absolute right-0 top-0 opacity-10 -mr-4 -mt-4 group-hover:opacity-20 transition-opacity"><DollarSign className="w-24 h-24 text-emerald-500" /></div>
            <div className="relative z-10">
                <p className="text-xs text-emerald-500/80 font-bold uppercase tracking-widest mb-1">Total Ganho</p>
                <div className="flex items-baseline gap-2">
                    <p className="text-3xl font-black text-emerald-400">+${stats.grossProfit.toFixed(2)}</p>
                </div>
                <p className="text-[10px] text-gray-500 mt-1 font-mono">Soma trades positivos</p>
            </div>
        </Card>

        {/* Total Perdido (R$) */}
        <Card className="glass-panel p-4 border-l-4 border-l-red-500 flex items-center justify-between relative overflow-hidden group hover:bg-red-500/5 transition-colors">
            <div className="absolute right-0 top-0 opacity-10 -mr-4 -mt-4 group-hover:opacity-20 transition-opacity"><DollarSign className="w-24 h-24 text-red-500" /></div>
            <div className="relative z-10">
                <p className="text-xs text-red-500/80 font-bold uppercase tracking-widest mb-1">Total Perdido</p>
                <div className="flex items-baseline gap-2">
                    <p className="text-3xl font-black text-red-400">${stats.grossLoss.toFixed(2)}</p>
                </div>
                <p className="text-[10px] text-gray-500 mt-1 font-mono">Soma trades negativos</p>
            </div>
        </Card>

        {/* Saldo Líquido (R$) */}
        <Card className={`glass-panel p-4 border-l-4 flex items-center justify-between relative overflow-hidden group transition-colors ${stats.totalPnl >= 0 ? 'border-l-amber-500 hover:bg-amber-500/5' : 'border-l-red-600 hover:bg-red-600/5'}`}>
            <div className={`absolute right-0 top-0 opacity-10 -mr-4 -mt-4 group-hover:opacity-20 transition-opacity`}>
                <TrendingUp className={`w-24 h-24 ${stats.totalPnl >= 0 ? 'text-amber-500' : 'text-red-500'}`} />
            </div>
            <div className="relative z-10">
                <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${stats.totalPnl >= 0 ? 'text-amber-500/80' : 'text-red-500/80'}`}>Resultado Líquido</p>
                <p className={`text-3xl font-black ${stats.totalPnl >= 0 ? "text-amber-400" : "text-red-400"}`}>
                    {stats.totalPnl >= 0 ? "+" : ""}{stats.totalPnl.toFixed(2)}
                </p>
                <p className="text-[10px] text-gray-500 mt-1 font-mono">Ganho - Perda</p>
            </div>
        </Card>

        {/* Estatística Geral */}
        <Card className="glass-panel p-4 border-l-4 border-l-primary flex items-center justify-between relative overflow-hidden group hover:bg-white/5 transition-colors">
            <div className="absolute right-0 top-0 opacity-10 -mr-4 -mt-4 group-hover:opacity-20 transition-opacity"><Activity className="w-24 h-24" /></div>
            <div className="relative z-10">
                <p className="text-xs text-gray-400 font-bold uppercase tracking-widest mb-1">Win Rate</p>
                <div className="flex items-baseline gap-2">
                    <p className="text-3xl font-black text-white">{stats.winRate.toFixed(0)}%</p>
                    <span className="text-xs text-gray-500 font-bold">({stats.total} trades)</span>
                </div>
            </div>
        </Card>
      </div>

      {/* Tabela Detalhada com Valores Monetários */}
      <Card className="glass-panel overflow-hidden border-red-900/30 flex flex-col h-[600px] relative">
        <div className="overflow-auto custom-scrollbar flex-1 w-full">
          <Table>
            <TableHeader className="bg-black/90 backdrop-blur-md sticky top-0 z-20 border-b border-white/10 shadow-lg">
              <TableRow className="border-white/5 hover:bg-transparent">
                <TableHead className="text-gray-400 font-bold text-xs uppercase w-[160px] pl-6 py-4">Data</TableHead>
                <TableHead className="text-gray-400 font-bold text-xs uppercase py-4">Ativo</TableHead>
                <TableHead className="text-gray-400 font-bold text-xs uppercase text-right py-4">Valor Investido</TableHead>
                <TableHead className="text-gray-400 font-bold text-xs uppercase text-right py-4">Valor Final</TableHead>
                <TableHead className="text-right text-gray-400 font-bold text-xs uppercase pr-6 py-4">Lucro/Prejuízo (R$)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trades?.slice().reverse().map((trade: any, idx: number) => {
                // Cálculos individuais para exibição na tabela
                const investido = trade.entry_price * trade.quantity;
                const valorFinal = trade.exit_price * trade.quantity;
                
                return (
                  <TableRow key={idx} className="border-white/5 hover:bg-white/5 transition-colors group">
                    <TableCell className="text-gray-400 font-mono text-xs font-medium pl-6">
                      {new Date(trade.timestamp).toLocaleString('pt-BR')}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                          <Badge variant="outline" className="bg-white/5 text-white border-white/10 font-bold group-hover:border-white/20 transition-colors">
                          {trade.symbol}
                          </Badge>
                          <Badge className={`px-2 py-0 text-[9px] font-bold uppercase tracking-wider ${trade.pnl > 0 
                              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" 
                              : "bg-red-500/10 text-red-400 border-red-500/20"}`}>
                            {trade.pnl > 0 ? "WIN" : "LOSS"}
                          </Badge>
                      </div>
                    </TableCell>
                    
                    {/* Coluna Valor Investido (Entrei com X) */}
                    <TableCell className="text-right font-mono text-sm text-gray-300">
                        ${investido ? investido.toFixed(2) : "-"}
                    </TableCell>

                    {/* Coluna Valor Final (Sai com Y) */}
                    <TableCell className="text-right font-mono text-sm text-gray-300">
                        ${valorFinal ? valorFinal.toFixed(2) : "-"}
                    </TableCell>

                    {/* Coluna Resultado (Ganhei/Perdi Z) */}
                    <TableCell className="text-right pr-6">
                      <div className="flex items-center justify-end gap-2">
                        <span className={`font-black font-mono text-sm ${trade.pnl > 0 ? "text-emerald-400" : "text-red-400"}`}>
                          {trade.pnl > 0 ? "+" : ""}{trade.pnl?.toFixed(2)}
                        </span>
                        {trade.pnl > 0 ? <ArrowUpRight className="w-4 h-4 text-emerald-500" /> : <ArrowDownRight className="w-4 h-4 text-red-500" />}
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
              {(!trades || trades.length === 0) && (
                <TableRow>
                  <TableCell colSpan={5} className="h-64 text-center">
                    <div className="flex flex-col items-center justify-center text-gray-500 opacity-50 gap-4">
                        <History className="w-16 h-16 stroke-1" />
                        <p className="font-medium text-lg">Nenhuma operação fechada ainda.</p>
                        <p className="text-sm max-w-sm mx-auto">Assim que o robô vender algo, os valores aparecerão aqui.</p>
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