import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowUpRight, ArrowDownRight, X } from "lucide-react";

const mockPositions = [
  {
    symbol: "BTC/USDT",
    entryPrice: 43250.50,
    currentPrice: 43890.25,
    quantity: 0.0231,
    invested: 998.89,
    currentValue: 1013.78,
    pnl: 14.89,
    pnlPct: 1.49,
    stopPrice: 42950.00,
    takePrice: 44550.00,
    openedAt: "2024-01-15 14:23:15"
  },
  {
    symbol: "ETH/USDT",
    entryPrice: 2456.80,
    currentPrice: 2489.50,
    quantity: 0.8124,
    invested: 1996.02,
    currentValue: 2023.37,
    pnl: 27.35,
    pnlPct: 1.37,
    stopPrice: 2432.00,
    takePrice: 2555.00,
    openedAt: "2024-01-15 15:10:42"
  },
];

const Positions = () => {
  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-6 bg-card border-border">
          <p className="text-sm text-muted-foreground mb-2">Posições Abertas</p>
          <p className="text-3xl font-bold text-foreground">{mockPositions.length}</p>
        </Card>
        <Card className="p-6 bg-card border-border">
          <p className="text-sm text-muted-foreground mb-2">Capital Investido</p>
          <p className="text-3xl font-bold text-foreground">
            ${mockPositions.reduce((acc, p) => acc + p.invested, 0).toFixed(2)}
          </p>
        </Card>
        <Card className="p-6 bg-card border-border">
          <p className="text-sm text-muted-foreground mb-2">Lucro Flutuante</p>
          <p className="text-3xl font-bold text-success">
            +${mockPositions.reduce((acc, p) => acc + p.pnl, 0).toFixed(2)}
          </p>
        </Card>
      </div>

      {/* Positions List */}
      <div className="space-y-4">
        {mockPositions.map((position, idx) => (
          <Card key={idx} className="p-6 bg-card border-border hover:border-primary/50 transition-all duration-300">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
              {/* Symbol & Entry */}
              <div className="space-y-2">
                <div className="flex items-center gap-3">
                  <h3 className="text-xl font-bold text-foreground">{position.symbol}</h3>
                  <Badge variant="outline" className="bg-primary/10 text-primary border-primary/30">
                    LONG
                  </Badge>
                </div>
                <div className="text-sm text-muted-foreground">
                  <p>Entrada: ${position.entryPrice.toFixed(2)} • Qtd: {position.quantity}</p>
                  <p className="text-xs">{position.openedAt}</p>
                </div>
              </div>

              {/* Price Info */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Preço Atual</p>
                  <p className="font-semibold text-foreground">${position.currentPrice.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Stop Loss</p>
                  <p className="font-semibold text-destructive">${position.stopPrice.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Take Profit</p>
                  <p className="font-semibold text-success">${position.takePrice.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">P&L</p>
                  <div className="flex items-center gap-1">
                    {position.pnl > 0 ? (
                      <ArrowUpRight className="w-4 h-4 text-success" />
                    ) : (
                      <ArrowDownRight className="w-4 h-4 text-destructive" />
                    )}
                    <span className={`font-semibold ${position.pnl > 0 ? "text-success" : "text-destructive"}`}>
                      ${Math.abs(position.pnl).toFixed(2)}
                    </span>
                    <span className={`text-xs ${position.pnl > 0 ? "text-success" : "text-destructive"}`}>
                      ({position.pnlPct > 0 ? "+" : ""}{position.pnlPct.toFixed(2)}%)
                    </span>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <Button variant="outline" size="sm" className="border-destructive/50 text-destructive hover:bg-destructive hover:text-destructive-foreground">
                <X className="w-4 h-4 mr-2" />
                Fechar
              </Button>
            </div>

            {/* Progress Bars */}
            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>Stop</span>
                <span>Entrada</span>
                <span>Take</span>
              </div>
              <div className="relative h-2 bg-secondary rounded-full overflow-hidden">
                {/* Calculate position percentage between stop and take */}
                <div 
                  className="absolute h-full bg-gradient-to-r from-destructive via-warning to-success"
                  style={{ 
                    width: `${((position.currentPrice - position.stopPrice) / (position.takePrice - position.stopPrice)) * 100}%` 
                  }}
                />
                {/* Current price indicator */}
                <div 
                  className="absolute top-0 w-1 h-full bg-foreground"
                  style={{ 
                    left: `${((position.currentPrice - position.stopPrice) / (position.takePrice - position.stopPrice)) * 100}%` 
                  }}
                />
              </div>
            </div>
          </Card>
        ))}

        {mockPositions.length === 0 && (
          <Card className="p-12 bg-card border-border text-center">
            <p className="text-muted-foreground">Nenhuma posição aberta no momento</p>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Positions;
