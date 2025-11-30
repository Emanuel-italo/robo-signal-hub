import { useState } from "react";
import Dashboard from "@/components/trading/Dashboard";
import Positions from "@/components/trading/Positions";
import TradeHistory from "@/components/trading/TradeHistory";
import Settings from "@/components/trading/Settings";
import LiveLogs from "@/components/trading/LiveLogs";
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
        {/* Sidebar usando variáveis do index.css */}
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
          <header className="flex h-16 items-center gap-4 px-6 border-b border-sidebar-border bg-background/50 backdrop-blur-sm sticky top-0 z-20">
            <SidebarTrigger className="text-foreground hover:bg-sidebar-accent" />
            <div className="flex-1">
              <h2 className="text-lg font-black text-foreground uppercase tracking-widest">
                {menuItems.find(i => i.id === activeView)?.title}
              </h2>
            </div>
            <div className="flex items-center gap-3 bg-card px-3 py-1.5 rounded-full border border-border shadow-sm">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-primary"></span>
              </span>
              <span className="text-xs font-bold text-primary">REDE NEURAL ATIVA</span>
            </div>
          </header>
          
          <main className="flex-1 p-6 overflow-auto">
            <div className="max-w-7xl mx-auto space-y-6">
              {renderContent()}
            </div>
          </main>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
};

export default Index;