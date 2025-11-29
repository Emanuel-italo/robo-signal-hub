import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";

const mockTrades = [
  {
    timestamp: "2024-01-15 13:45:22",
    symbol: "BTC/USDT",
    type: "TAKE PROFIT 💰",
    entryPrice: 42890.50,
    exitPrice: 43456.20,
    quantity: 0.0233,
    pnl: 13.18,
    pnlPct: 1.32
  },
  {
    timestamp: "2024-01-15 12:30:15",
    symbol: "ETH/USDT",
    type: "STOP LOSS 🛑",
    entryPrice: 2467.80,
    exitPrice: 2443.20,
    quantity: 0.8124,
    pnl: -19.98,
    pnlPct: -0.99
  },
  {
    timestamp: "2024-01-15 11:20:45",
    symbol: "SOL/USDT",
    type: "TAKE PROFIT 💰",
    entryPrice: 98.45,
    exitPrice: 100.23,
    quantity: 10.15,
    pnl: 18.06,
    pnlPct: 1.81
  },
  {
    timestamp: "2024-01-15 10:15:30",
    symbol: "BNB/USDT",
    type: "TAKE PROFIT 💰",
    entryPrice: 312.80,
    exitPrice: 318.45,
    quantity: 3.195,
    pnl: 18.04,
    pnlPct: 1.81
  },
  {
    timestamp: "2024-01-15 09:05:12",
    symbol: "XRP/USDT",
    type: "STOP LOSS 🛑",
    entryPrice: 0.5834,
    exitPrice: 0.5775,
    quantity: 1715.5,
    pnl: -10.12,
    pnlPct: -1.01
  },
];

const TradeHistory = () => {
  const totalPnL = mockTrades.reduce((acc, t) => acc + t.pnl, 0);
  const winningTrades = mockTrades.filter(t => t.pnl > 0).length;
  const winRate = (winningTrades / mockTrades.length) * 100;

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-6 bg-card border-border">
          <p className="text-sm text-muted-foreground mb-2">Total de Trades</p>
          <p className="text-3xl font-bold text-foreground">{mockTrades.length}</p>
        </Card>
        <Card className="p-6 bg-card border-border">
          <p className="text-sm text-muted-foreground mb-2">Trades Ganhos</p>
          <p className="text-3xl font-bold text-success">{winningTrades}</p>
        </Card>
        <Card className="p-6 bg-card border-border">
          <p className="text-sm text-muted-foreground mb-2">Win Rate</p>
          <p className="text-3xl font-bold text-success">{winRate.toFixed(1)}%</p>
        </Card>
        <Card className="p-6 bg-card border-border">
          <p className="text-sm text-muted-foreground mb-2">P&L Total</p>
          <p className={`text-3xl font-bold ${totalPnL > 0 ? "text-success" : "text-destructive"}`}>
            {totalPnL > 0 ? "+" : ""}${totalPnL.toFixed(2)}
          </p>
        </Card>
      </div>

      {/* Trade Table */}
      <Card className="bg-card border-border overflow-hidden">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-border hover:bg-transparent">
                <TableHead className="text-muted-foreground">Data/Hora</TableHead>
                <TableHead className="text-muted-foreground">Símbolo</TableHead>
                <TableHead className="text-muted-foreground">Tipo</TableHead>
                <TableHead className="text-muted-foreground">Entrada</TableHead>
                <TableHead className="text-muted-foreground">Saída</TableHead>
                <TableHead className="text-muted-foreground">Quantidade</TableHead>
                <TableHead className="text-muted-foreground text-right">P&L</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {mockTrades.map((trade, idx) => (
                <TableRow key={idx} className="border-border hover:bg-secondary/30">
                  <TableCell className="text-foreground font-medium">{trade.timestamp}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className="bg-accent/10 text-accent border-accent/30">
                      {trade.symbol}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge 
                      variant={trade.type.includes("PROFIT") ? "default" : "destructive"}
                      className={trade.type.includes("PROFIT") ? "bg-success/20 text-success border-success/30" : "bg-destructive/20 text-destructive border-destructive/30"}
                    >
                      {trade.type}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-foreground">${trade.entryPrice.toFixed(2)}</TableCell>
                  <TableCell className="text-foreground">${trade.exitPrice.toFixed(2)}</TableCell>
                  <TableCell className="text-muted-foreground">{trade.quantity}</TableCell>
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-2">
                      {trade.pnl > 0 ? (
                        <ArrowUpRight className="w-4 h-4 text-success" />
                      ) : (
                        <ArrowDownRight className="w-4 h-4 text-destructive" />
                      )}
                      <span className={`font-semibold ${trade.pnl > 0 ? "text-success" : "text-destructive"}`}>
                        {trade.pnl > 0 ? "+" : ""}${trade.pnl.toFixed(2)}
                      </span>
                      <span className={`text-sm ${trade.pnl > 0 ? "text-success" : "text-destructive"}`}>
                        ({trade.pnlPct > 0 ? "+" : ""}{trade.pnlPct.toFixed(2)}%)
                      </span>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Card>
    </div>
  );
};

export default TradeHistory;
