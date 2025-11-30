import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/services/api";
import Dashboard from "@/components/trading/Dashboard";
import Positions from "@/components/trading/Positions";
import TradeHistory from "@/components/trading/TradeHistory";
import Settings from "@/components/trading/Settings";
import LiveLogs from "@/components/trading/LiveLogs";
import MarketTicker from "@/components/trading/MarketTicker";
import { 
  LayoutDashboard, 
  TrendingUp, 
  History, 
  Terminal, 
  Settings as SettingsIcon,
  Zap
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  SidebarInset
} from "@/components/ui/sidebar";

const Index = () => {
  const [activeView, setActiveView] = useState("dashboard");

  // O Index busca o status apenas para atualizar o indicador "SISTEMA ONLINE" no header
  const { data: status } = useQuery({
    queryKey: ["botStatus"],
    queryFn: api.getStatus,
    refetchInterval: 2000,
  });

  const renderContent = () => {
    switch (activeView) {
      case "dashboard": return <Dashboard />;
      case "positions": return <Positions />;
      case "history": return <TradeHistory />;
      case "logs": return <LiveLogs />;
      case "settings": return <Settings />;
      default: return <Dashboard />;
    }
  };

  const menuItems = [
    { title: "DASHBOARD", id: "dashboard", icon: LayoutDashboard },
    { title: "POSIÇÕES", id: "positions", icon: TrendingUp },
    { title: "HISTÓRICO", id: "history", icon: History },
    { title: "LOGS AO VIVO", id: "logs", icon: Terminal },
    { title: "SISTEMA", id: "settings", icon: SettingsIcon },
  ];

  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full bg-background/50">
        <Sidebar className="border-r border-sidebar-border">
          <SidebarContent>
            <div className="p-6 flex items-center gap-3">
              <div className="w-10 h-10 bg-primary rounded-xl flex items-center justify-center shadow-lg shadow-red-500/20 animate-float">
                <Zap className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="font-black text-xl tracking-tighter text-foreground">RED.SIGNAL</h1>
                <p className="text-[10px] font-bold text-primary tracking-widest uppercase">AI TRADING V1</p>
              </div>
            </div>
            
            <SidebarGroup>
              <SidebarGroupContent>
                <SidebarMenu className="space-y-2 px-2">
                  {menuItems.map((item) => (
                    <SidebarMenuItem key={item.id}>
                      <SidebarMenuButton 
                        isActive={activeView === item.id}
                        onClick={() => setActiveView(item.id)}
                        className={`h-12 rounded-xl transition-all duration-300 font-bold tracking-wide
                          ${activeView === item.id 
                            ? "bg-primary text-white shadow-lg shadow-red-900/40 translate-x-1 hover:bg-primary/90 hover:text-white" 
                            : "text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground hover:translate-x-1"
                          }`}
                      >
                        <item.icon className={`w-5 h-5 ${activeView === item.id ? "animate-pulse" : ""}`} />
                        <span>{item.title}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>
        </Sidebar>

        <SidebarInset className="flex flex-col min-w-0 bg-transparent">
          {/* Header Fixo com Letreiro */}
          <div className="sticky top-0 z-20 flex flex-col">
             
             {/* CORREÇÃO AQUI: Removemos "prices={...}" pois o componente já busca os dados sozinho */}
             <div className="w-full bg-black/80 backdrop-blur border-b border-white/10">
                <MarketTicker />
             </div>
             
             {/* Barra de Navegação */}
             <header className="flex h-14 items-center gap-4 px-6 border-b border-sidebar-border bg-background/80 backdrop-blur-md">
                <SidebarTrigger className="text-foreground hover:bg-sidebar-accent" />
                <div className="flex-1">
                <h2 className="text-lg font-black text-foreground uppercase tracking-widest">
                    {menuItems.find(i => i.id === activeView)?.title}
                </h2>
                </div>
                <div className="flex items-center gap-3 bg-card px-3 py-1.5 rounded-full border border-border shadow-sm">
                <span className="relative flex h-2.5 w-2.5">
                    <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${status?.isRunning ? "bg-emerald-500" : "bg-red-500"}`}></span>
                    <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${status?.isRunning ? "bg-emerald-500" : "bg-red-500"}`}></span>
                </span>
                <span className={`text-xs font-bold ${status?.isRunning ? "text-emerald-500" : "text-red-500"}`}>
                    {status?.isRunning ? "SISTEMA ONLINE" : "SISTEMA PAUSADO"}
                </span>
                </div>
            </header>
          </div>
          
          <main className="flex-1 p-6 overflow-auto">
            <div className="max-w-7xl mx-auto space-y-6 pb-20">
              {renderContent()}
            </div>
          </main>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
};

export default Index;