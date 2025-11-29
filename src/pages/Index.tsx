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
  LineChart
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  SidebarInset
} from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";

const Index = () => {
  const [activeView, setActiveView] = useState("dashboard");

  // Mapeamento das telas
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
    { title: "Visão Geral", id: "dashboard", icon: LayoutDashboard },
    { title: "Posições Abertas", id: "positions", icon: TrendingUp },
    { title: "Histórico", id: "history", icon: History },
    { title: "Logs do Sistema", id: "logs", icon: Terminal },
    { title: "Configurações", id: "settings", icon: SettingsIcon },
  ];

  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full bg-background">
        <Sidebar>
          <SidebarContent>
            <div className="p-4 flex items-center gap-2 font-bold text-xl text-primary">
              <div className="w-8 h-8 bg-primary text-primary-foreground rounded flex items-center justify-center">
                <LineChart className="w-5 h-5" />
              </div>
              CryptoTrader AI
            </div>
            <Separator className="mb-4 opacity-50" />
            
            <SidebarGroup>
              <SidebarGroupLabel>Menu Principal</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {menuItems.map((item) => (
                    <SidebarMenuItem key={item.id}>
                      <SidebarMenuButton 
                        isActive={activeView === item.id}
                        onClick={() => setActiveView(item.id)}
                        className="h-10"
                      >
                        <item.icon className="w-4 h-4" />
                        <span>{item.title}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>
        </Sidebar>

        <SidebarInset className="flex flex-col min-w-0">
          <header className="flex h-14 items-center gap-4 border-b bg-card/50 px-4 backdrop-blur-sm sticky top-0 z-10">
            <SidebarTrigger />
            <div className="flex-1 font-medium">
              {menuItems.find(i => i.id === activeView)?.title}
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="w-2 h-2 bg-success rounded-full animate-pulse" />
              Sistema Online
            </div>
          </header>
          
          <main className="flex-1 p-6 overflow-auto">
            <div className="max-w-6xl mx-auto space-y-6">
              {renderContent()}
            </div>
          </main>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
};

export default Index;