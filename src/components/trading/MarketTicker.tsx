import { TrendingUp, TrendingDown } from "lucide-react";

interface MarketTickerProps {
  prices: Record<string, number>;
}

const MarketTicker = ({ prices }: MarketTickerProps) => {
  const symbols = Object.entries(prices);

  if (symbols.length === 0) return null;

  return (
    <div className="w-full overflow-hidden bg-black/80 border-b border-red-900/30 h-8 flex items-center relative z-50">
      <div className="flex gap-8 animate-marquee whitespace-nowrap px-4">
        {/* Duplicado para efeito de loop visual */}
        {[...symbols, ...symbols].map(([symbol, price], idx) => (
          <div key={`${symbol}-${idx}`} className="flex items-center gap-2 min-w-fit">
            <span className="font-bold text-[10px] text-gray-500 uppercase tracking-wider">{symbol.replace('/USDT','')}</span>
            <div className="flex items-center gap-1">
              <span className="font-mono text-xs font-black text-white">${price.toFixed(2)}</span>
              {Math.random() > 0.5 ? (
                <TrendingUp className="w-3 h-3 text-emerald-500" />
              ) : (
                <TrendingDown className="w-3 h-3 text-red-500" />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MarketTicker;