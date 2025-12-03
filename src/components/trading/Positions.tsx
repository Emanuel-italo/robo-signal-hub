import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowUpRight, ArrowDownRight, X, Layers, Activity, Loader2 } from "lucide-react";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
import { useState } from "react";

const Positions = () => {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [loadingSymbol, setLoadingSymbol] = useState<string | null>(null);

  const { data: status } = useQuery({
    queryKey: ["botStatus"],
    queryFn: api.getStatus,
    refetchInterval: 2000, 
  });

  // --- LÓGICA DE VENDA MANUAL ---
  const handleManualSell = async (symbol: string) => {
    try {
      setLoadingSymbol(symbol); // Ativa o spinner no botão
      toast({
        title: "Enviando ordem...",
        description: `Solicitando venda a mercado para ${symbol}.`,
      });

      // Chama o backend
      const result = await api.closePosition(symbol);

      if (result && result.status === 'success') {
        toast({
          title: "Ordem Enviada!",
          description: `Posição de ${symbol} encerrada com sucesso.`,
          variant: "default",
          className: "bg-emerald-500 text-white border-none",
        });
        // Força uma atualização imediata da lista para o ativo sumir da tela
        await queryClient.invalidateQueries({ queryKey: ["botStatus"] });
      } else {
        throw new Error(result?.message || "Erro desconhecido ao vender.");
      }
    } catch (error: any) {
      console.error("Erro venda manual:", error);
      toast({
        title: "Erro na Venda",
        description: error.response?.data?.message || "Falha ao comunicar com o robô. Verifique se ele está rodando.",
        variant: "destructive",
      });
    } finally {
      setLoadingSymbol(null); // Desativa o spinner
    }
  };
  // -----------------------------

  const positions = status?.openPositions || [];

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      
      {/* Cabeçalho da Seção */}
      <div className="flex items-center gap-4 mb-8">
        <div className="p-3 bg-red-950/30 border border-red-900/50 rounded-xl">
          <Layers className="w-6 h-6 text-primary animate-pulse" />
        </div>
        <div>
          <h2 className="text-2xl font-black text-white tracking-wide">POSIÇÕES ABERTAS</h2>
          <p className="text-gray-500 font-medium">Gerenciamento de risco ativo</p>
        </div>
      </div>

      {positions.length === 0 ? (
        <Card className="glass-panel p-12 text-center border-dashed border-2 border-red-900/30 flex flex-col items-center justify-center gap-4">
          <div className="w-16 h-16 rounded-full bg-black/50 flex items-center justify-center mb-2">
            <Activity className="w-8 h-8 text-gray-600" />
          </div>
          <h3 className="text-xl font-bold text-white">Nenhuma Posição Ativa</h3>
          <p className="text-gray-500 max-w-sm">
            O robô está escaneando o mercado em busca de oportunidades com alta probabilidade.
          </p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {positions.map((pos: any, idx: number) => (
            <Card key={idx} className="glass-panel p-6 group hover:border-red-500/50 transition-all duration-300 relative overflow-hidden">
              <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <Layers className="w-32 h-32 text-red-500" />
              </div>

              <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
                
                {/* Info do Ativo */}
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-full bg-red-950/50 border border-red-900/50 flex items-center justify-center font-bold text-xs text-red-500">
                    {pos.symbol.substring(0,3)}
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-xl font-black text-white tracking-wider">{pos.symbol}</h3>
                      <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500/20 font-bold">
                        LONG
                      </Badge>
                    </div>
                    <div className="flex gap-4 text-xs text-gray-400 font-mono">
                      <span>QTD: <span className="text-white">{pos.quantity}</span></span>
                      <span>ENTRADA: <span className="text-white">${pos.entryPrice}</span></span>
                    </div>
                  </div>
                </div>

                {/* Métricas e PnL */}
                <div className="flex-1 grid grid-cols-2 lg:grid-cols-4 gap-4 bg-black/40 p-4 rounded-xl border border-white/5">
                  <div>
                    <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">Valor Investido</p>
                    <p className="font-mono text-white">${pos.invested?.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">Preço Atual</p>
                    <p className="font-mono text-white">${pos.currentPrice?.toFixed(2) || "..."}</p>
                  </div>
                  <div className="col-span-2 lg:col-span-2">
                    <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">Lucro / Prejuízo (PnL)</p>
                    <div className="flex items-center gap-2">
                      {(pos.pnl || 0) >= 0 ? (
                        <ArrowUpRight className="w-5 h-5 text-emerald-500" />
                      ) : (
                        <ArrowDownRight className="w-5 h-5 text-red-500" />
                      )}
                      <span className={`text-xl font-black ${(pos.pnl || 0) >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                        ${(pos.pnl || 0).toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Ações (Botão Vender Agora) */}
                <Button 
                  variant="outline" 
                  className="border-red-900/50 text-red-500 hover:bg-red-950 hover:text-red-400 hover:border-red-500 transition-all cursor-pointer min-w-[140px]"
                  onClick={() => handleManualSell(pos.symbol)}
                  disabled={loadingSymbol === pos.symbol}
                >
                  {loadingSymbol === pos.symbol ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" /> VENDENDO...
                    </>
                  ) : (
                    <>
                      <X className="w-4 h-4 mr-2" /> VENDER AGORA
                    </>
                  )}
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default Positions;