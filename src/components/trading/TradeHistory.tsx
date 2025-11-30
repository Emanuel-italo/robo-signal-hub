import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ArrowUpRight, ArrowDownRight, History } from "lucide-react";
import { api } from "@/services/api";

const TradeHistory = () => {
  const { data: trades } = useQuery({
    queryKey: ["botHistory"],
    queryFn: api.getHistory,
    refetchInterval: 5000,
  });

  return (
    <div className="space-y-6 animate-in fade-in duration-700">
      <div className="flex items-center gap-4 mb-6">
        <div className="p-3 bg-red-950/30 border border-red-900/50 rounded-xl">
          <History className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h2 className="text-2xl font-black text-white tracking-wide">HISTÓRICO DE EXECUÇÃO</h2>
          <p className="text-gray-500 font-medium">Registro imutável de operações</p>
        </div>
      </div>

      <Card className="glass-panel overflow-hidden border-red-900/30">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader className="bg-white/5">
              <TableRow className="border-white/5 hover:bg-transparent">
                <TableHead className="text-gray-400 font-bold text-xs uppercase">Data</TableHead>
                <TableHead className="text-gray-400 font-bold text-xs uppercase">Par</TableHead>
                <TableHead className="text-gray-400 font-bold text-xs uppercase">Tipo</TableHead>
                <TableHead className="text-gray-400 font-bold text-xs uppercase">Preço Ent.</TableHead>
                <TableHead className="text-gray-400 font-bold text-xs uppercase">Preço Saída</TableHead>
                <TableHead className="text-right text-gray-400 font-bold text-xs uppercase">Resultado (PnL)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trades?.slice().reverse().map((trade: any, idx: number) => (
                <TableRow key={idx} className="border-white/5 hover:bg-white/5 transition-colors group">
                  <TableCell className="text-gray-300 font-mono text-xs">{trade.timestamp}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className="bg-transparent text-white border-white/20 font-bold">
                      {trade.symbol}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge className={trade.pnl > 0 ? "bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30" : "bg-red-500/20 text-red-400 hover:bg-red-500/30"}>
                      {trade.tipo || (trade.pnl > 0 ? "WIN" : "LOSS")}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-gray-400 font-mono">${trade.entry_price?.toFixed(2)}</TableCell>
                  <TableCell className="text-gray-400 font-mono">${trade.exit_price?.toFixed(2)}</TableCell>
                  <TableCell className="text-right">
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
                  <TableCell colSpan={6} className="h-32 text-center text-gray-500">
                    Nenhum registro encontrado no histórico.
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