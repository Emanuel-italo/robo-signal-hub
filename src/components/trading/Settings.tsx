import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Save, Sliders, ShieldAlert, Cpu } from "lucide-react";

const Settings = () => {
  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-right-8 duration-700">
      
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-red-950/30 border border-red-900/50 rounded-xl">
            <Sliders className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h2 className="text-2xl font-black text-white tracking-wide">SISTEMA E PARAMETRIZAÇÃO</h2>
            <p className="text-gray-500 font-medium">Ajuste fino dos algoritmos</p>
          </div>
        </div>
        <Button className="bg-primary hover:bg-red-600 text-white font-bold shadow-lg shadow-red-900/20">
          <Save className="w-4 h-4 mr-2" /> SALVAR ALTERAÇÕES
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Card Trading */}
        <Card className="glass-panel p-6 space-y-6">
          <div className="flex items-center gap-2 border-b border-white/5 pb-4 mb-4">
            <Cpu className="w-5 h-5 text-red-400" />
            <h3 className="font-bold text-white uppercase tracking-wider">Parâmetros de Execução</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label className="text-gray-400 text-xs uppercase font-bold">Capital Inicial (Simulado)</Label>
              <Input type="number" defaultValue="10000" className="bg-black/50 border-red-900/30 text-white focus:border-red-500 font-mono" />
            </div>
            <div className="space-y-2">
              <Label className="text-gray-400 text-xs uppercase font-bold">Modo de Investimento</Label>
              <Select defaultValue="fixed">
                <SelectTrigger className="bg-black/50 border-red-900/30 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-black border-red-900 text-white">
                  <SelectItem value="fixed">Fração Fixa (%)</SelectItem>
                  <SelectItem value="risk">Baseado em Risco</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label className="text-gray-400 text-xs uppercase font-bold">Stop Loss %</Label>
              <Input type="number" defaultValue="2.0" className="bg-black/50 border-red-900/30 text-white focus:border-red-500 font-mono" />
            </div>
            <div className="space-y-2">
              <Label className="text-gray-400 text-xs uppercase font-bold">Take Profit %</Label>
              <Input type="number" defaultValue="4.0" className="bg-black/50 border-red-900/30 text-white focus:border-red-500 font-mono" />
            </div>
          </div>
        </Card>

        {/* Card Risco */}
        <Card className="glass-panel p-6 space-y-6">
          <div className="flex items-center gap-2 border-b border-white/5 pb-4 mb-4">
            <ShieldAlert className="w-5 h-5 text-red-400" />
            <h3 className="font-bold text-white uppercase tracking-wider">Gestão de Risco</h3>
          </div>

          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label className="text-gray-400 text-xs uppercase font-bold">Confiança Mínima da IA</Label>
                <span className="text-red-400 font-mono text-xs font-bold">60%</span>
              </div>
              <div className="h-2 bg-black/50 rounded-full overflow-hidden">
                <div className="h-full w-[60%] bg-gradient-to-r from-red-900 to-red-500"></div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label className="text-gray-400 text-xs uppercase font-bold">Max Trades / Dia</Label>
                <Input type="number" defaultValue="5" className="bg-black/50 border-red-900/30 text-white focus:border-red-500 font-mono" />
              </div>
              <div className="space-y-2">
                <Label className="text-gray-400 text-xs uppercase font-bold">Exposição Max (%)</Label>
                <Input type="number" defaultValue="25" className="bg-black/50 border-red-900/30 text-white focus:border-red-500 font-mono" />
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Settings;