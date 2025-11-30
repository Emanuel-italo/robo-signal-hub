import { TrendingUp, TrendingDown, Bitcoin } from "lucide-react";

interface MarketTickerProps {
  prices?: Record<string, number>;
}

const MarketTicker = ({ prices }: MarketTickerProps) => {
  // Ajuste: Forçamos o tipo "as [string, number][]" para o TypeScript entender que são pares corretos
  const symbols: [string, number][] = prices 
    ? Object.entries(prices) 
    : [["BTC/USDT", 0], ["ETH/USDT", 0]];
  
  // Duplicar a lista para o efeito de marquee infinito
  const displaySymbols = [...symbols, ...symbols, ...symbols, ...symbols];

  return (
    <div className="w-full bg-black/90 border-b border-red-900/50 h-10 flex items-center overflow-hidden relative z-50 shadow-lg shadow-red-900/20">
      {/* Ícone fixo à esquerda */}
      <div className="absolute left-0 z-20 bg-black/90 h-full px-3 flex items-center border-r border-red-900/50">
        <Bitcoin className="w-5 h-5 text-primary animate-pulse" />
      </div>

      {/* Faixa de Gradiente para suavizar a entrada/saída */}
      <div className="absolute left-10 z-10 w-8 h-full bg-gradient-to-r from-black to-transparent pointer-events-none" />
      <div className="absolute right-0 z-10 w-8 h-full bg-gradient-to-l from-black to-transparent pointer-events-none" />

      {/* Conteúdo Animado */}
      <div className="flex animate-marquee whitespace-nowrap pl-12">
        {displaySymbols.map(([symbol, price], idx) => {
            const isPositive = Math.random() > 0.4;
            return (
            <div key={`${symbol}-${idx}`} className="flex items-center gap-2 mx-6">
                <span className="font-black text-xs text-red-500/80 uppercase tracking-widest">
                {symbol.replace('/USDT','')}
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