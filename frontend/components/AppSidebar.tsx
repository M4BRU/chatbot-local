"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Bot, FileText, MessageSquare, Mic, Plus, Trash2 } from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { useConversation } from "@/app/providers";

const MODES = [
  { label: "Chat", href: "/", icon: MessageSquare },
  { label: "Devis", href: "/devis", icon: FileText },
  { label: "Agent", href: "/agent", icon: Bot },
  { label: "Transcription", href: "/transcription", icon: Mic },
];

export function AppSidebar() {
  const pathname = usePathname();
  const { conversations, currentConversationId, selectConversation, createConversation, deleteConversation, mode } =
    useConversation();

  const handleNewConversation = async () => {
    await createConversation("Nouvelle conversation", mode);
  };

  return (
    <Sidebar>
      <SidebarHeader className="p-4 space-y-3">
        <div className="flex items-center gap-2">
          <Image
            src="/logoVLM.png"
            alt="VLM Robotics"
            width={120}
            height={40}
            className="object-contain"
          />
        </div>
        <Button
          onClick={handleNewConversation}
          className="w-full"
          size="sm"
        >
          <Plus className="h-4 w-4 mr-2" />
          Nouvelle conversation
        </Button>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>MODE</SidebarGroupLabel>
          <SidebarMenu>
            {MODES.map(({ label, href, icon: Icon }) => (
              <SidebarMenuItem key={href}>
                <SidebarMenuButton asChild isActive={pathname === href}>
                  <Link href={href}>
                    <Icon className="h-4 w-4" />
                    {label}
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>HISTORIQUE</SidebarGroupLabel>
          <SidebarMenu>
            {conversations.length === 0 && (
              <p className="px-2 py-1 text-xs text-sidebar-foreground/50">Aucune conversation</p>
            )}
            {conversations.map((conv) => (
              <SidebarMenuItem key={conv.id}>
                <SidebarMenuButton
                  isActive={conv.id === currentConversationId}
                  onClick={() => selectConversation(conv.id)}
                  className="flex justify-between group"
                >
                  <span className="truncate flex-1 text-left">{conv.title}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteConversation(conv.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 transition-opacity ml-2 shrink-0"
                    title="Supprimer"
                  >
                    <Trash2 className="h-3 w-3 text-muted-foreground hover:text-destructive" />
                  </button>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="p-4">
        <Link
          href="/admin"
          className="text-sm text-sidebar-foreground/70 hover:text-sidebar-foreground transition-colors"
        >
          Admin
        </Link>
      </SidebarFooter>
    </Sidebar>
  );
}
