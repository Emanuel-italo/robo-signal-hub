import { useState, useEffect, useMemo, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { 
  Wallet, DollarSign, Activity, Power, TrendingUp, Target, BrainCircuit, 
  ShieldCheck, Filter, ArrowUpRight, ArrowDownRight, Zap, Volume2, VolumeX,
  X, Loader2, Coins, ShoppingCart
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from "recharts";
// Mantendo importações relativas para evitar erros de build
import { api, BotStatus } from "../../services/api"; 
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { useWakeLock } from "../../hooks/use-wake-lock";
import MarketTicker from "./MarketTicker";

interface ChartData {
  time: string;
  value: number;
}

const COLORS = ['#10b981', '#ef4444']; // Emerald (Win) vs Red (Loss)

const Dashboard = () => {
  // --- ESTADOS E DADOS ---
  const queryClient = useQueryClient();
  const { data: rawStatus, refetch: refetchStatus } = useQuery({
    queryKey: ["botStatus"],
    queryFn: api.getStatus,
    refetchInterval: 1000, 
  });

  const { data: history } = useQuery({
    queryKey: ["botHistory"],
    queryFn: api.getHistory,
    refetchInterval: 5000,
  });

  const status = rawStatus as BotStatus | undefined;
  
  const { requestLock, releaseLock } = useWakeLock();
  
  const [equityData, setEquityData] = useState<ChartData[]>([]);
  const [winRateData, setWinRateData] = useState<{name: string, value: number}[]>([]);
  const [selectedAsset, setSelectedAsset] = useState<string>("GLOBAL");
  
  // Controle de Som
  const [soundEnabled, setSoundEnabled] = useState(true);
  
  // Controle de loading para venda manual
  const [loadingSymbol, setLoadingSymbol] = useState<string | null>(null);

  // --- NOVOS ESTADOS PARA COMPRA MANUAL ---
  const [buySymbol, setBuySymbol] = useState("");
  const [buyAmount, setBuyAmount] = useState("10"); // Valor padrão $10
  const [isBuying, setIsBuying] = useState(false);

  // Referências para controle de áudio e eventos (CORREÇÃO DO LOOP/DURAÇÃO)
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastProcessedEventRef = useRef<string | null>(null);

  // --- SISTEMA DE ÁUDIO INTELIGENTE ---
  const playSound = (type: 'buy' | 'win' | 'loss') => {
    if (!soundEnabled) return;

    // 1. Limpa qualquer som que esteja tocando antes de começar o novo
    if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
    }
    if (audioTimeoutRef.current) {
        clearTimeout(audioTimeoutRef.current);
    }

    try {
        const audio = new Audio(`/sounds/${type}.mp3`);
        audio.volume = 0.5;
        audioRef.current = audio;

        const playPromise = audio.play();
        if (playPromise !== undefined) {
            playPromise.catch(e => console.log("Erro ao reproduzir (interação necessária ou arquivo ausente):", e));
        }

        // 2. Trava de Segurança: Para o som EXATAMENTE após 11 segundos
        audioTimeoutRef.current = setTimeout(() => {
            if (audioRef.current === audio) { // Garante que estamos parando o som certo
                audio.pause();
                audio.currentTime = 0;
            }
        }, 11000); // 11000ms = 11 segundos

    } catch (error) {
        console.error("Audio error", error);
    }
  };

  // --- LÓGICA DE VENDA MANUAL ---
  const handleManualSell = async (symbol: string) => {
    try {
      setLoadingSymbol(symbol); // Ativa o spinner no botão
      toast.info(`Enviando ordem de venda para ${symbol}...`);

      // Chama o backend
      const result = await api.closePosition(symbol);

      if (result && result.status === 'success') {
        toast.success(`Ordem enviada! Posição de ${symbol} encerrada.`);
        // Força uma atualização imediata da lista para o ativo sumir da tela
        await queryClient.invalidateQueries({ queryKey: ["botStatus"] });
      } else {
        throw new Error(result?.message || "Erro desconhecido ao vender.");
      }
    } catch (error: any) {
      console.error("Erro venda manual:", error);
      toast.error(error.response?.data?.message || "Falha ao comunicar com o robô.");
    } finally {
      setLoadingSymbol(null); // Desativa o spinner
    }
  };

  // --- LÓGICA DE COMPRA MANUAL ---
  const handleManualBuy = async () => {
    if (!buySymbol) {
      toast.error("Digite o símbolo do ativo (ex: BTC)");
      return;
    }

    try {
      setIsBuying(true);
      toast.info(`Comprando ${buySymbol.toUpperCase()}...`);
      
      // Remove espaços e garante /USDT se o usuário esqueceu
      let sym = buySymbol.toUpperCase().trim();
      if (!sym.includes("/")) sym += "/USDT";

      const result = await api.openPosition(sym, parseFloat(buyAmount));

      if (result && result.status === 'success') {
        toast.success(`Compra realizada com sucesso!`, {
            description: `${sym} adicionado ao portfólio.`
        });
        setBuySymbol(""); // Limpa o input
        playSound('buy'); // Toca o som de compra
        await queryClient.invalidateQueries({ queryKey: ["botStatus"] }); // Atualiza tela
      }
    } catch (error: any) {
      console.error("Erro compra manual:", error);
      toast.error(error.response?.data?.message || "Falha ao comprar.");
    } finally {
      setIsBuying(false);
    }
  };

  // --- EFEITOS (NOTIFICAÇÕES E SOM) ---
  useEffect(() => {
    if (status?.lastEvent) {
      const evt = status.lastEvent;
      
      // Assinatura única baseada no conteúdo do evento
      const eventSignature = JSON.stringify(evt);

      // Só processa se for um evento novo
      if (eventSignature !== lastProcessedEventRef.current) {
          lastProcessedEventRef.current = eventSignature;

          if (evt.type === 'BUY') {
            playSound('buy');
            toast.success(`COMPRA EXECUTADA: ${evt.symbol}`, {
              description: `Entrada em $${evt.price?.toFixed(2)}`,
              className: "bg-black border-2 border-red-500/50 text-white font-bold"
            });
          } else if (evt.type === 'SELL') {
            const isWin = (evt.pnl || 0) > 0;
            playSound(isWin ? 'win' : 'loss');
            toast(isWin ? "LUCRO REALIZADO!" : "STOP LOSS", {
              description: `${evt.symbol} PnL: $${evt.pnl?.toFixed(2)}`,
              className: isWin ? "bg-emerald-950 border-emerald-500 text-emerald-400" : "bg-red-950 border-red-500 text-red-400"
            });
          }
      }
    }
  }, [status?.lastEvent]);

  // Wake Lock
  useEffect(() => {
    if (status?.isRunning) requestLock();
    else releaseLock();
  }, [status?.isRunning, requestLock, releaseLock]);

  // Atualiza Gráfico de Patrimônio
  useEffect(() => {
    if (status?.equity) {
      const now = new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      setEquityData(prev => {
        if (prev.length > 0 && prev[prev.length - 1].time === now) return prev;
        const newData = [...prev, { time: now, value: status.equity }];
        return newData.slice(-40); 
      });
    }
  }, [status?.equity]);

  // Atualiza Dados de Win Rate
  useEffect(() => {
    if (history && Array.isArray(history)) {
      const wins = history.filter((t: any) => t.pnl > 0).length;
      const losses = history.filter((t: any) => t.pnl <= 0).length;
      
      if (wins + losses > 0) {
        setWinRateData([
          { name: 'Wins', value: wins },
          { name: 'Losses', value: losses }
        ]);
      } else {
        setWinRateData([]);
      }
    }
  }, [history]);

  const handleToggleBot = async () => {
    try {
      if (status?.isRunning) {
        await api.stopBot();
      } else {
        await api.startBot();
        playSound('buy'); // Feedback sonoro ao iniciar
      }
      setTimeout(() => refetchStatus(), 500);
    } catch (error) {
      console.error("Erro toggle:", error);
      toast.error("Erro de conexão com o robô");
    }
  };

  // --- CÁLCULOS FINANCEIROS DETALHADOS (LUCRO vs PREJUÍZO) ---
  const metrics = useMemo(() => {
    if (!status) return null;

    const positions = status.openPositions || [];
    let baseMetrics;

    // Métricas Básicas (Topo)
    if (selectedAsset === "GLOBAL") {
        const totalInvested = positions.reduce((acc, pos) => acc + pos.invested, 0);
        const totalPnL = positions.reduce((acc, pos) => acc + pos.pnl, 0);
        
        baseMetrics = {
            title: "PATRIMÔNIO TOTAL (SE VENDER TUDO)", // Rótulo Corrigido
            value: status.equity,
            subValue: status.balance,
            subLabel: "SALDO LIVRE (PARA GASTAR)", // Rótulo Corrigido
            pnl: totalPnL,
            pnlPercent: totalInvested > 0 ? (totalPnL / totalInvested) * 100 : 0,
            isGlobal: true
        };
    } else {
        const pos = positions.find(p => p.symbol === selectedAsset);
        if (!pos) return null;

        baseMetrics = {
            title: `POSIÇÃO: ${pos.symbol}`,
            value: pos.invested + pos.pnl,
            subValue: pos.invested,
            subLabel: "Valor Investido",
            pnl: pos.pnl,
            pnlPercent: (pos.pnl / pos.invested) * 100,
            isGlobal: false
        };
    }

    // Métricas Detalhadas de Histórico (Ganhos vs Perdas Totais)
    let grossProfit = 0;
    let grossLoss = 0;

    if (history && Array.isArray(history)) {
        const filteredHistory = selectedAsset === "GLOBAL" 
            ? history 
            : history.filter((t: any) => t.symbol === selectedAsset);

        grossProfit = filteredHistory
            .filter((t: any) => t.pnl > 0)
            .reduce((acc: number, t: any) => acc + t.pnl, 0);

        grossLoss = filteredHistory
            .filter((t: any) => t.pnl < 0)
            .reduce((acc: number, t: any) => acc + t.pnl, 0);
    }

    return { ...baseMetrics, grossProfit, grossLoss };
  }, [status, selectedAsset, history]);

  // Cálculo da Taxa de Acerto para exibição central
  const winRatePercent = useMemo(() => {
    if (winRateData.length === 0) return 0;
    const wins = winRateData.find(d => d.name === 'Wins')?.value || 0;
    const total = winRateData.reduce((acc, curr) => acc + curr.value, 0);
    return total > 0 ? (wins / total) * 100 : 0;
  }, [winRateData]);

  // Reset do filtro se o ativo sumir
  useEffect(() => {
      const positions = status?.openPositions || [];
      if (status && selectedAsset !== "GLOBAL" && !positions.find(p => p.symbol === selectedAsset)) {
          setSelectedAsset("GLOBAL");
      }
  }, [status, selectedAsset]);

  if (!metrics) return (
    <div className="flex h-screen items-center justify-center bg-black">
        <div className="text-center space-y-4">
            <Activity className="w-12 h-12 text-red-500 animate-spin mx-auto" />
            <p className="text-white font-bold animate-pulse">Carregando Sistemas...</p>
        </div>
    </div>
  );

  return (
    <div className="space-y-6 animate-in fade-in zoom-in-95 duration-700 pb-10">

      {/* --- TICKER --- */}
      {status && status.marketPrices && (
        <div className="rounded-xl border-y border-red-900/30 overflow-hidden shadow-lg bg-black/60 backdrop-blur-md">
           <MarketTicker prices={status.marketPrices} />
        </div>
      )}
      
      {/* --- HERO BANNER --- */}
      <div className="relative overflow-hidden rounded-3xl p-1">
        <div className="absolute inset-0 bg-gradient-to-r from-red-600 via-red-900 to-black opacity-50 blur-sm"></div>
        
        <div className="relative bg-black/80 backdrop-blur-xl rounded-[22px] p-6 md:p-8 flex flex-col md:flex-row justify-between items-center gap-6 overflow-hidden">
            <div className="absolute top-0 right-0 -mt-20 -mr-20 w-96 h-96 bg-red-600/10 rounded-full blur-[100px]"></div>
            
            <div className="flex items-center gap-5 z-10">
                <div className="relative">
                    <div className={`absolute inset-0 bg-red-500 blur-xl opacity-20 ${status?.isRunning ? 'animate-pulse' : ''}`}></div>
                    <div className="p-4 bg-gradient-to-br from-red-950 to-black border border-red-800/50 rounded-2xl shadow-inner relative">
                        <BrainCircuit className={`w-10 h-10 ${status?.isRunning ? 'text-red-400' : 'text-gray-500'}`} />
                    </div>
                </div>
                <div>
                    <h2 className="text-3xl md:text-4xl font-black tracking-tighter text-white italic">
                        PAINEL <span className="text-red-500">COMANDO</span>
                    </h2>
                    <div className="flex items-center gap-3 mt-1">
                        <div className={`h-2 w-2 rounded-full ${status?.isRunning ? 'bg-emerald-500 animate-pulse shadow-[0_0_10px_#10b981]' : 'bg-red-500'}`}></div>
                        <p className="text-gray-400 font-mono text-sm uppercase tracking-widest">
                            {status?.isRunning ? "Sistema Online" : "Sistema Parado"}
                        </p>
                    </div>
                </div>
            </div>
            
            <div className="flex gap-3 relative z-10">
                 <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setSoundEnabled(!soundEnabled)}
                    className={`h-14 w-14 rounded-xl border-white/10 bg-black/40 hover:bg-white/10 transition-colors ${soundEnabled ? 'text-emerald-400' : 'text-red-400'}`}
                 >
                    {soundEnabled ? <Volume2 className="w-6 h-6" /> : <VolumeX className="w-6 h-6" />}
                 </Button>

                <Button 
                    size="lg"
                    onClick={handleToggleBot}
                    className={`
                        relative h-14 px-10 text-lg font-bold tracking-widest uppercase transition-all duration-500
                        overflow-hidden group border
                        ${status?.isRunning 
                            ? "bg-red-500/10 text-red-500 border-red-500/50 hover:bg-red-500 hover:text-white hover:border-red-500" 
                            : "bg-emerald-500/10 text-emerald-500 border-emerald-500/50 hover:bg-emerald-500 hover:text-white hover:border-emerald-500"
                        }
                    `}
                >
                    <Power className={`w-5 h-5 mr-3 transition-transform group-hover:scale-125 ${status?.isRunning ? "animate-pulse" : ""}`} />
                    {status?.isRunning ? "Desativar" : "Iniciar"}
                </Button>
            </div>
        </div>
      </div>

      {/* --- CONTROL BAR E VENDA MANUAL --- */}
      <div className="glass-panel p-4 rounded-xl flex flex-col md:flex-row gap-4 items-center justify-between border-l-4 border-l-red-600">
        <div className="flex items-center gap-3 w-full md:w-auto">
            <div className="p-2 bg-white/5 rounded-lg">
                <Filter className="w-5 h-5 text-red-400" />
            </div>
            <div className="flex-1 md:flex-none">
                <span className="text-[10px] text-gray-500 font-bold uppercase block mb-1">Filtrar Ativo</span>
                <Select value={selectedAsset} onValueChange={setSelectedAsset}>
                    <SelectTrigger className="w-full md:w-[280px] bg-black/40 border-white/10 text-white font-bold h-10 focus:ring-red-500/50">
                        <SelectValue placeholder="Selecione..." />
                    </SelectTrigger>
                    <SelectContent className="bg-[#0a0a0a] border-red-900/50 text-white">
                        <SelectItem value="GLOBAL" className="font-black text-red-500 py-3">⚡ PORTFÓLIO GLOBAL</SelectItem>
                        {status?.openPositions?.map((pos) => (
                            <SelectItem key={pos.symbol} value={pos.symbol} className="font-mono text-gray-300">
                                {pos.symbol}
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
            </div>
        </div>

        {/* --- NOVO BLOCO: COMPRA MANUAL --- */}
        <div className="flex items-center gap-2 border-l border-white/10 pl-4 ml-4">
            <div className="flex flex-col">
                <span className="text-[10px] text-emerald-500 font-bold uppercase block mb-1">
                    Compra Rápida ($)
                </span>
                <div className="flex items-center gap-2">
                    <Input 
                        placeholder="Ativo (ex: ETH)" 
                        className="w-24 h-10 bg-black/40 border-emerald-500/30 text-white font-bold placeholder:text-gray-600 focus:border-emerald-500"
                        value={buySymbol}
                        onChange={(e) => setBuySymbol(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleManualBuy()}
                    />
                    <Input 
                        type="number"
                        placeholder="$" 
                        className="w-16 h-10 bg-black/40 border-emerald-500/30 text-white font-bold focus:border-emerald-500 text-center"
                        value={buyAmount}
                        onChange={(e) => setBuyAmount(e.target.value)}
                    />
                    <Button 
                        size="icon"
                        className="h-10 w-10 bg-emerald-600 hover:bg-emerald-500 text-white"
                        onClick={handleManualBuy}
                        disabled={isBuying}
                    >
                        {isBuying ? <Loader2 className="w-4 h-4 animate-spin" /> : <ShoppingCart className="w-4 h-4" />}
                    </Button>
                </div>
            </div>
        </div>

        <div className="flex items-center gap-6 divide-x divide-white/10 w-full md:w-auto justify-end">
            {/* BOTÃO DE VENDA MANUAL NO FILTRO */}
            {selectedAsset !== "GLOBAL" && (
                <div className="pr-6">
                    <Button 
                        variant="destructive"
                        className="font-bold border border-red-500/30 hover:border-red-500 bg-red-950/50 hover:bg-red-900 transition-all"
                        onClick={() => handleManualSell(selectedAsset)}
                        disabled={loadingSymbol === selectedAsset}
                    >
                        {loadingSymbol === selectedAsset ? (
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        ) : (
                            <X className="w-4 h-4 mr-2" />
                        )}
                        VENDER {selectedAsset}
                    </Button>
                </div>
            )}

            <div className="pl-6 text-right">
                <p className="text-sm font-bold text-gray-500 uppercase tracking-widest mb-1">Resultado Aberto (PnL)</p>
                <div className={`flex items-center justify-end gap-3 text-5xl font-black font-mono tracking-tighter ${metrics.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {metrics.pnl >= 0 ? "+" : ""}{metrics.pnl.toFixed(2)}
                    {metrics.pnl >= 0 ? <ArrowUpRight className="w-8 h-8 text-emerald-500" /> : <ArrowDownRight className="w-8 h-8 text-red-500" />}
                </div>
            </div>
        </div>
      </div>

      {/* --- KPI GRID --- */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* CARD 1: PATRIMÔNIO (EQUITY) */}
        <Card className="glass-panel p-6 relative overflow-hidden group hover:border-red-500/40 transition-all duration-300">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <TrendingUp className="w-20 h-20 text-red-500" />
          </div>
          <div>
            <p className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">{metrics.title}</p>
            <h3 className="text-4xl font-black text-white tracking-tight">${metrics.value.toFixed(2)}</h3>
            <p className="text-[10px] text-gray-400 mt-1">Soma do Caixa + Valor das Posições</p>
          </div>
          <div className="h-12 w-full mt-4 opacity-40 group-hover:opacity-60 transition-opacity">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={equityData.slice(-15)}>
                <Line type="monotone" dataKey="value" stroke="#ef4444" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* CARD 2: CAIXA (BALANCE) */}
        <Card className="glass-panel p-6 relative overflow-hidden group hover:border-emerald-500/40 transition-all duration-300">
           <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Wallet className="w-20 h-20 text-emerald-500" />
          </div>
          <div className="flex flex-col h-full justify-between relative z-10">
            <div>
              <p className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">{metrics.subLabel}</p>
              <h3 className="text-4xl font-black text-white tracking-tight">${metrics.subValue.toFixed(2)}</h3>
              <p className="text-[10px] text-gray-400 mt-1">Disponível para novas compras</p>
            </div>
            <div className="mt-6 flex items-center gap-2">
                <div className="p-2 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                    <DollarSign className="w-4 h-4 text-emerald-400" />
                </div>
                <span className="text-sm text-gray-400 font-medium">Líquido na Binance</span>
            </div>
          </div>
        </Card>

        {/* CARD 3: META & WINRATE */}
        <Card className="glass-panel p-1 flex items-center justify-between hover:border-amber-500/40 transition-all duration-300">
           <div className="flex-1 p-5 h-full flex flex-col justify-center">
              <div className="flex items-center gap-2 mb-2">
                 <Target className="w-4 h-4 text-amber-500" />
                 <p className="text-xs font-bold text-gray-500 uppercase tracking-widest">Meta Diária</p>
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-4xl font-black text-white">{status?.dailyTrades || 0}</span>
                <span className="text-sm font-bold text-gray-600">/ 10</span>
              </div>
              <div className="w-full bg-white/5 h-2 rounded-full mt-3 overflow-hidden">
                <div 
                    className="bg-amber-500 h-full transition-all duration-1000" 
                    style={{ width: `${Math.min(((status?.dailyTrades || 0) / 10) * 100, 100)}%` }}
                ></div>
              </div>
           </div>
           <div className="w-32 h-32 relative mr-2">
             <ResponsiveContainer width="100%" height="100%">
               <PieChart>
                 <Pie
                   data={winRateData.length > 0 ? winRateData : [{name: 'Empty', value: 1}]}
                   innerRadius={35}
                   outerRadius={50}
                   dataKey="value"
                   stroke="none"
                   startAngle={90}
                   endAngle={-270}
                 >
                   {winRateData.map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                   ))}
                   {winRateData.length === 0 && <Cell fill="#333" />}
                 </Pie>
               </PieChart>
             </ResponsiveContainer>
             <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                <span className="text-xs font-bold text-gray-500">WIN RATE</span>
                <span className={`text-lg font-black ${winRatePercent >= 50 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {winRatePercent.toFixed(0)}%
                </span>
             </div>
           </div>
        </Card>
      </div>

      {/* --- DETALHAMENTO FINANCEIRO (PROFIT VS LOSS) --- */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* CARD LUCRO BRUTO */}
        <Card className="glass-panel p-6 border-l-4 border-l-emerald-500 flex items-center justify-between relative overflow-hidden hover:bg-emerald-500/5 transition-colors group">
            <div className="absolute right-0 top-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity"><ArrowUpRight className="w-32 h-32 text-emerald-500" /></div>
            <div className="relative z-10">
                <p className="text-xs font-bold text-emerald-500/80 uppercase tracking-widest mb-1">Total Ganho (Gross Profit)</p>
                <h3 className="text-3xl font-black text-emerald-400">
                    +${metrics.grossProfit.toFixed(2)}
                </h3>
                <p className="text-xs text-gray-500 mt-2 font-medium">Soma de todas as operações vencedoras</p>
            </div>
            <div className="p-4 bg-emerald-500/10 rounded-full border border-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.2)]">
                <DollarSign className="w-8 h-8 text-emerald-400" />
            </div>
        </Card>

        {/* CARD PREJUÍZO BRUTO */}
        <Card className="glass-panel p-6 border-l-4 border-l-red-500 flex items-center justify-between relative overflow-hidden hover:bg-red-500/5 transition-colors group">
            <div className="absolute right-0 top-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity"><ArrowDownRight className="w-32 h-32 text-red-500" /></div>
            <div className="relative z-10">
                <p className="text-xs font-bold text-red-500/80 uppercase tracking-widest mb-1">Total Perdido (Gross Loss)</p>
                <h3 className="text-3xl font-black text-red-400">
                    ${metrics.grossLoss.toFixed(2)}
                </h3>
                <p className="text-xs text-gray-500 mt-2 font-medium">Soma de todas as operações perdedoras</p>
            </div>
            <div className="p-4 bg-red-500/10 rounded-full border border-red-500/20 shadow-[0_0_15px_rgba(239,68,68,0.2)]">
                <Activity className="w-8 h-8 text-red-400" />
            </div>
        </Card>

        {/* CARD SALDO LÍQUIDO (NOVO) */}
        <Card className={`glass-panel p-6 border-l-4 flex items-center justify-between relative overflow-hidden transition-colors group ${
            (metrics.grossProfit + metrics.grossLoss) >= 0 ? 'border-l-emerald-400 hover:bg-emerald-500/5' : 'border-l-red-500 hover:bg-red-500/5'
        }`}>
            <div className="absolute right-0 top-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <Coins className={`w-32 h-32 ${(metrics.grossProfit + metrics.grossLoss) >= 0 ? 'text-emerald-500' : 'text-red-500'}`} />
            </div>
            <div className="relative z-10">
                <p className={`text-xs font-bold uppercase tracking-widest mb-1 ${(metrics.grossProfit + metrics.grossLoss) >= 0 ? 'text-emerald-500/80' : 'text-red-500/80'}`}>
                    Resultado Líquido (Net PnL)
                </p>
                <h3 className={`text-3xl font-black ${(metrics.grossProfit + metrics.grossLoss) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {(metrics.grossProfit + metrics.grossLoss) >= 0 ? '+' : ''}${(metrics.grossProfit + metrics.grossLoss).toFixed(2)}
                </h3>
                <p className="text-xs text-gray-500 mt-2 font-medium">Ganho Real (Lucro - Prejuízo)</p>
            </div>
            <div className={`p-4 rounded-full border shadow-[0_0_15px_rgba(0,0,0,0.2)] ${(metrics.grossProfit + metrics.grossLoss) >= 0 
                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' 
                : 'bg-red-500/10 border-red-500/20 text-red-400'}`}>
                <Coins className="w-8 h-8" />
            </div>
        </Card>
      </div>

      {/* --- GRÁFICO PRINCIPAL --- */}
      <Card className="glass-panel p-6 border-red-500/10 shadow-[0_0_50px_rgba(0,0,0,0.5)]">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-red-600 to-black flex items-center justify-center shadow-lg border border-red-500/30">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-black text-white tracking-wide">
                {selectedAsset === "GLOBAL" ? "CURVA DE CRESCIMENTO" : `PERFORMANCE: ${selectedAsset}`}
              </h3>
              <div className="flex items-center gap-2 mt-1">
                 <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                </span>
                <p className="text-xs font-bold text-red-400">DADOS EM TEMPO REAL</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="h-[450px] w-full relative">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={equityData}>
              <defs>
                <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#dc2626" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#dc2626" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" vertical={false} />
              <XAxis 
                dataKey="time" 
                stroke="#525252" 
                fontSize={10} 
                tickLine={false}
                axisLine={false}
                minTickGap={40}
                fontWeight={700}
                dy={10}
              />
              <YAxis 
                domain={['auto', 'auto']} 
                stroke="#525252" 
                fontSize={10} 
                tickLine={false}
                axisLine={false}
                tickFormatter={(val) => `$${val}`}
                fontWeight={700}
                dx={-10}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: "#000", 
                  border: "1px solid #333", 
                  borderRadius: "8px",
                  boxShadow: "0 10px 30px rgba(0,0,0,0.8)",
                  fontWeight: "bold",
                  color: "#fff"
                }}
                itemStyle={{ color: "#ef4444" }}
                labelStyle={{ color: "#6b7280", marginBottom: "0.25rem", fontSize: "0.7rem", textTransform: "uppercase" }}
                cursor={{ stroke: '#ef4444', strokeWidth: 1, strokeDasharray: '5 5' }}
              />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="#ef4444" 
                strokeWidth={3} 
                fill="url(#colorEquity)" 
                animationDuration={1500}
                activeDot={{ r: 6, fill: "#ef4444", stroke: "#fff", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
          
          {equityData.length === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-black/20 backdrop-blur-sm rounded-xl">
              <div className="p-6 bg-black border border-white/10 rounded-2xl shadow-2xl text-center">
                <Zap className="w-10 h-10 text-red-500 animate-pulse mx-auto mb-3" />
                <p className="font-bold text-white text-lg">Sincronizando Feed...</p>
                <p className="text-gray-500 text-xs mt-1">Aguardando dados da Binance</p>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;