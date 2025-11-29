import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Save, AlertCircle } from "lucide-react";

const Settings = () => {
  return (
    <div className="space-y-6">
      {/* Trading Settings */}
      <Card className="p-6 bg-card border-border">
        <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-warning" />
          Configurações de Trading
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <Label htmlFor="capital" className="text-foreground">Capital Inicial (USD)</Label>
            <Input id="capital" type="number" defaultValue="10000" className="bg-secondary border-border text-foreground" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="risk" className="text-foreground">Risco por Trade (%)</Label>
            <Input id="risk" type="number" defaultValue="1" step="0.1" className="bg-secondary border-border text-foreground" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="investment_mode" className="text-foreground">Modo de Investimento</Label>
            <Select defaultValue="fixed_fraction">
              <SelectTrigger className="bg-secondary border-border text-foreground">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="fixed_fraction">Fixed Fraction</SelectItem>
                <SelectItem value="risk_based">Risk Based</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="fixed_fraction" className="text-foreground">Fixed Fraction (%)</Label>
            <Input id="fixed_fraction" type="number" defaultValue="2" step="0.1" className="bg-secondary border-border text-foreground" />
          </div>
        </div>
      </Card>

      {/* Stop Loss & Take Profit */}
      <Card className="p-6 bg-card border-border">
        <h3 className="text-lg font-semibold text-foreground mb-4">Stop Loss & Take Profit</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <Label htmlFor="atr_stop" className="text-foreground">ATR Stop Multiplier</Label>
            <Input id="atr_stop" type="number" defaultValue="1.0" step="0.1" className="bg-secondary border-border text-foreground" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="atr_take" className="text-foreground">ATR Take Multiplier</Label>
            <Input id="atr_take" type="number" defaultValue="2.0" step="0.1" className="bg-secondary border-border text-foreground" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="stop_pct" className="text-foreground">Stop Loss % (Fallback)</Label>
            <Input id="stop_pct" type="number" defaultValue="2" step="0.1" className="bg-secondary border-border text-foreground" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="take_pct" className="text-foreground">Take Profit % (Fallback)</Label>
            <Input id="take_pct" type="number" defaultValue="4" step="0.1" className="bg-secondary border-border text-foreground" />
          </div>
        </div>
      </Card>

      {/* Risk Management */}
      <Card className="p-6 bg-card border-border">
        <h3 className="text-lg font-semibold text-foreground mb-4">Gestão de Risco</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <Label htmlFor="max_daily_trades" className="text-foreground">Máximo de Trades Diários</Label>
            <Input id="max_daily_trades" type="number" defaultValue="5" className="bg-secondary border-border text-foreground" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="max_exposure" className="text-foreground">Exposição Máxima (%)</Label>
            <Input id="max_exposure" type="number" defaultValue="25" className="bg-secondary border-border text-foreground" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="min_confidence" className="text-foreground">Confiança Mínima (%)</Label>
            <Input id="min_confidence" type="number" defaultValue="60" className="bg-secondary border-border text-foreground" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="min_oos_acc" className="text-foreground">Acurácia OOS Mínima (%)</Label>
            <Input id="min_oos_acc" type="number" defaultValue="55" className="bg-secondary border-border text-foreground" />
          </div>
        </div>
      </Card>

      {/* Model Settings */}
      <Card className="p-6 bg-card border-border">
        <h3 className="text-lg font-semibold text-foreground mb-4">Configurações do Modelo</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <Label htmlFor="lookahead" className="text-foreground">Lookahead Candles</Label>
            <Input id="lookahead" type="number" defaultValue="24" className="bg-secondary border-border text-foreground" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="retrain_interval" className="text-foreground">Intervalo de Retreino (minutos)</Label>
            <Input id="retrain_interval" type="number" defaultValue="1440" className="bg-secondary border-border text-foreground" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="vwap_window" className="text-foreground">VWAP Window (candles)</Label>
            <Input id="vwap_window" type="number" defaultValue="24" className="bg-secondary border-border text-foreground" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="timeframe" className="text-foreground">Timeframe</Label>
            <Select defaultValue="1h">
              <SelectTrigger className="bg-secondary border-border text-foreground">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="15m">15 minutos</SelectItem>
                <SelectItem value="30m">30 minutos</SelectItem>
                <SelectItem value="1h">1 hora</SelectItem>
                <SelectItem value="4h">4 horas</SelectItem>
                <SelectItem value="1d">1 dia</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </Card>

      {/* System Settings */}
      <Card className="p-6 bg-card border-border">
        <h3 className="text-lg font-semibold text-foreground mb-4">Configurações do Sistema</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg">
            <div>
              <p className="font-medium text-foreground">Modo Live Trading</p>
              <p className="text-sm text-muted-foreground">Ativar trading real na exchange (CUIDADO!)</p>
            </div>
            <Switch defaultChecked={false} />
          </div>
          <div className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg">
            <div>
              <p className="font-medium text-foreground">Warm Start</p>
              <p className="text-sm text-muted-foreground">Usar warm start no treinamento do modelo</p>
            </div>
            <Switch defaultChecked={false} />
          </div>
        </div>
      </Card>

      {/* Symbols */}
      <Card className="p-6 bg-card border-border">
        <h3 className="text-lg font-semibold text-foreground mb-4">Símbolos Ativos</h3>
        <div className="space-y-2">
          {["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"].map((symbol) => (
            <div key={symbol} className="flex items-center justify-between p-3 bg-secondary/30 rounded-lg">
              <span className="font-medium text-foreground">{symbol}</span>
              <Switch defaultChecked={true} />
            </div>
          ))}
        </div>
      </Card>

      {/* Save Button */}
      <div className="flex justify-end">
        <Button size="lg" className="gap-2">
          <Save className="w-4 h-4" />
          Salvar Configurações
        </Button>
      </div>
    </div>
  );
};

export default Settings;
