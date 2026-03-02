"use client";

import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import {
  createConversation as apiCreateConversation,
  deleteConversation as apiDeleteConversation,
  fetchConversations,
  updateConversationTitle,
} from "@/app/lib/api";
import type { Conversation } from "@/app/lib/types";

interface ConversationContextType {
  conversations: Conversation[];
  currentConversationId: string | null;
  mode: "chat" | "devis" | "transcription";
  setMode: (mode: "chat" | "devis" | "transcription") => void;
  selectConversation: (id: string) => void;
  createConversation: (title: string, mode: string) => Promise<string>;
  deleteConversation: (id: string) => void;
  refreshConversations: () => void;
  updateTitle: (id: string, title: string) => void;
}

const ConversationContext = createContext<ConversationContextType | null>(null);

export function useConversation() {
  const ctx = useContext(ConversationContext);
  if (!ctx) throw new Error("useConversation must be used within ConversationProvider");
  return ctx;
}

export function ConversationProvider({ children }: { children: React.ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [mode, setMode] = useState<"chat" | "devis" | "transcription">("chat");

  const refreshConversations = useCallback(async () => {
    const data = await fetchConversations();
    setConversations(data);
  }, []);

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  const selectConversation = useCallback((id: string) => {
    setCurrentConversationId(id);
  }, []);

  const createConversation = useCallback(async (title: string, convMode: string): Promise<string> => {
    const conv = await apiCreateConversation(title, convMode);
    setConversations((prev) => [conv, ...prev]);
    setCurrentConversationId(conv.id);
    return conv.id;
  }, []);

  const deleteConversation = useCallback(async (id: string) => {
    await apiDeleteConversation(id);
    setConversations((prev) => prev.filter((c) => c.id !== id));
    setCurrentConversationId((prev) => (prev === id ? null : prev));
  }, []);

  const updateTitle = useCallback(async (id: string, title: string) => {
    await updateConversationTitle(id, title);
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title } : c))
    );
  }, []);

  return (
    <ConversationContext.Provider
      value={{
        conversations,
        currentConversationId,
        mode,
        setMode,
        selectConversation,
        createConversation,
        deleteConversation,
        refreshConversations,
        updateTitle,
      }}
    >
      {children}
    </ConversationContext.Provider>
  );
}
