import { useEffect, useState } from "react";
import { TrendingUp, TrendingDown, Bitcoin, Activity } from "lucide-react";

interface MarketTickerProps {
  prices?: Record<string, number>;
}

const MarketTicker = ({ prices: externalPrices }: MarketTickerProps) => {
  const [internalPrices, setInternalPrices] = useState<[string, number][]>([]);

  useEffect(() => {
    // Se já vieram preços do robô, usa eles
    if (externalPrices && Object.keys(externalPrices).length > 0) {
      setInternalPrices(Object.entries(externalPrices));
      return;
    }

    // SENÃO: Busca preços públicos da Binance (Modo Independente)
    const fetchPrices = async () => {
      try {
        // Busca top moedas para não pesar
        const symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"];
        const response = await fetch("https://api.binance.com/api/v3/ticker/price");
        const data = await response.json();
        
        const filtered = data
          .filter((item: any) => symbols.includes(item.symbol))
          .map((item: any) => [
            item.symbol.replace("USDT", "/USDT"), 
            parseFloat(item.price)
          ]) as [string, number][];
          
        setInternalPrices(filtered);
      } catch (error) {
        // Fallback silencioso se der erro na API pública
        console.error("Erro ticker:", error);
      }
    };

    fetchPrices();
    const interval = setInterval(fetchPrices, 10000); // Atualiza a cada 10s
    return () => clearInterval(interval);

  }, [externalPrices]);

  // Se não tiver nada, mostra loading ou placeholder
  if (internalPrices.length === 0) return (
    <div className="w-full h-10 bg-black/90 flex items-center justify-center text-red-500/50 text-xs font-mono animate-pulse">
        <Activity className="w-3 h-3 mr-2" /> CARREGANDO MERCADO...
    </div>
  );

  // Duplica lista para efeito infinito
  const displaySymbols = [...internalPrices, ...internalPrices, ...internalPrices, ...internalPrices];

  return (
    <div className="w-full bg-black/90 border-b border-red-900/50 h-10 flex items-center overflow-hidden relative z-50 shadow-lg shadow-red-900/20">
      <div className="absolute left-0 z-20 bg-black/90 h-full px-3 flex items-center border-r border-red-900/50">
        <Bitcoin className="w-5 h-5 text-primary animate-pulse" />
      </div>

      <div className="absolute left-10 z-10 w-8 h-full bg-gradient-to-r from-black to-transparent pointer-events-none" />
      <div className="absolute right-0 z-10 w-8 h-full bg-gradient-to-l from-black to-transparent pointer-events-none" />

      <div className="flex animate-marquee whitespace-nowrap pl-12">
        {displaySymbols.map(([symbol, price], idx) => {
            const isPositive = Math.random() > 0.4; // Simulação visual de tendência
            return (
            <div key={`${symbol}-${idx}`} className="flex items-center gap-2 mx-6">
                <span className="font-black text-xs text-red-500/80 uppercase tracking-widest">
                {symbol}
                </span>
                <div className="flex items-center gap-1">
                <span className={`font-mono text-sm font-bold ${price > 0 ? "text-white" : "text-gray-500"}`}>
                    ${price > 0 ? price.toLocaleString('en-US', { minimumFractionDigits: 2 }) : "---"}
                </span>
                {price > 0 && (
                    isPositive ? 
                    <TrendingUp className="w-3 h-3 text-emerald-500" /> : 
                    <TrendingDown className="w-3 h-3 text-red-500" />
                )}
                </div>
            </div>
            )
        })}
      </div>
    </div>
  );
};

export default MarketTicker;