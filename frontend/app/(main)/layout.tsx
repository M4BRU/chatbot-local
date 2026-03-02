import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { ConversationProvider } from "@/app/providers";

export default function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <ConversationProvider>
      <SidebarProvider>
        <AppSidebar />
        <main className="flex-1 overflow-hidden flex flex-col">
          {children}
        </main>
      </SidebarProvider>
    </ConversationProvider>
  );
}
