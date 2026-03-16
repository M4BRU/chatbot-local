"use client";

import { useCallback, useRef, useState, useEffect } from "react";
import { Bot, Brain, Database, Send, Zap } from "lucide-react";
import { fetchCollections, streamAgentChat } from "@/app/lib/api";
import type { AgentMode, AgentStep, ChatSource } from "@/app/lib/types";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { AssistantMessage, LoadingDots, UserBubble } from "@/components/chat/MarkdownMessage";

// ─── Agent mode labels & colors ──────────────────────────────────────────────

const AGENT_META: Record<AgentMode, { label: string; color: string; icon: React.ReactNode }> = {
  simple_claude: {
    label: "Claude API",
    color: "bg-violet-500/10 border-violet-500/20 text-violet-600 dark:text-violet-400",
    icon: <Brain className="h-3 w-3" />,
  },
  simple_gpt: {
    label: "GPT-4o",
    color: "bg-green-500/10 border-green-500/20 text-green-600 dark:text-green-400",
    icon: <Brain className="h-3 w-3" />,
  },
  simple_rag: {
    label: "VLM Docs",
    color: "bg-blue-500/10 border-blue-500/20 text-blue-600 dark:text-blue-400",
    icon: <Database className="h-3 w-3" />,
  },
  combined: {
    label: "Combined",
    color: "bg-emerald-500/10 border-emerald-500/20 text-emerald-600 dark:text-emerald-400",
    icon: <Zap className="h-3 w-3" />,
  },
  combined_gpt: {
    label: "Combined GPT",
    color: "bg-teal-500/10 border-teal-500/20 text-teal-600 dark:text-teal-400",
    icon: <Zap className="h-3 w-3" />,
  },
};

const STEP_LABELS: Record<AgentStep | "forced", string> = {
  classifying: "Analyse de la question…",
  forced: "Mode forcé",
  claude_query: "Consultation Claude API…",
  gpt_query: "Consultation GPT-4o…",
  rag_search: "Recherche dans les docs VLM…",
  synthesizing: "Synthèse locale…",
};

// ─── Step indicator ───────────────────────────────────────────────────────────

function StepBadge({ step }: { step: AgentStep | "forced" }) {
  return (
    <div className="flex items-center gap-2 text-xs rounded-full px-3 py-1.5 w-fit mb-2 border bg-amber-500/10 border-amber-500/20 text-amber-600 dark:text-amber-400">
      <svg className="animate-spin h-3 w-3 shrink-0" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
      </svg>
      {STEP_LABELS[step] ?? step}
    </div>
  );
}

// ─── Agent badge ──────────────────────────────────────────────────────────────

function AgentBadge({ agent }: { agent: AgentMode }) {
  const meta = AGENT_META[agent];
  return (
    <div className={cn("flex items-center gap-1.5 text-xs rounded-full px-3 py-1 w-fit mb-2 border", meta.color)}>
      {meta.icon}
      {meta.label}
    </div>
  );
}

// ─── Message types ────────────────────────────────────────────────────────────

interface AgentMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  agent?: AgentMode;
  sources?: ChatSource[];
  isStreaming?: boolean;
  currentStep?: AgentStep | "forced";
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function AgentPage() {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [collections, setCollections] = useState<string[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string>("default");
  const [forceMode, setForceMode] = useState<AgentMode | "auto">("auto");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    fetchCollections().then((cols) => {
      if (cols.length > 0) {
        setCollections(cols);
        setSelectedCollection(cols[0]);
      }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = useCallback(async () => {
    const msg = input.trim();
    if (!msg || isLoading) return;

    const userMsg: AgentMessage = { id: crypto.randomUUID(), role: "user", content: msg };
    const assistantId = crypto.randomUUID();
    const assistantMsg: AgentMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const mode = forceMode === "auto" ? undefined : forceMode;
      for await (const event of streamAgentChat(msg, selectedCollection, mode)) {
        if (event.error) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: `Erreur : ${event.error}`, isStreaming: false, currentStep: undefined }
                : m
            )
          );
          break;
        }
        if (event.agent_step) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, currentStep: event.agent_step } : m
            )
          );
        }
        if (event.agent) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, agent: event.agent, currentStep: undefined } : m
            )
          );
        }
        if (event.token) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + event.token } : m
            )
          );
        }
        if (event.done) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, isStreaming: false, currentStep: undefined, sources: event.sources }
                : m
            )
          );
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: `Erreur réseau : ${String(err)}`, isStreaming: false, currentStep: undefined }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, selectedCollection, forceMode]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b bg-background shrink-0">
        <SidebarTrigger />
        <Bot className="h-5 w-5 text-emerald-500" />
        <span className="font-semibold">Agent Orchestrateur</span>
        <span className="text-xs text-muted-foreground ml-1">
          Claude · GPT-4o · VLM Docs
        </span>

        {/* Collection selector */}
        {collections.length > 0 && (
          <select
            value={selectedCollection}
            onChange={(e) => setSelectedCollection(e.target.value)}
            className="ml-auto text-xs border rounded px-2 py-1 bg-background"
          >
            {collections.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        )}

        {/* Force mode toggle */}
        <select
          value={forceMode}
          onChange={(e) => setForceMode(e.target.value as AgentMode | "auto")}
          className="text-xs border rounded px-2 py-1 bg-background"
          title="Forcer le mode"
        >
          <option value="auto">🤖 Auto</option>
          <option value="simple_claude">🧠 Claude API</option>
          <option value="simple_gpt">🟢 GPT-4o</option>
          <option value="simple_rag">📚 VLM Docs</option>
          <option value="combined">⚡ Combined (Claude)</option>
          <option value="combined_gpt">⚡ Combined (GPT)</option>
        </select>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-[680px] w-full mx-auto px-4 md:px-6 py-6 md:py-10">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground gap-3">
              <Bot className="h-12 w-12 opacity-20" />
              <p className="text-sm max-w-sm">
                Posez n&apos;importe quelle question. L&apos;agent choisit automatiquement
                entre Claude API (questions générales) et les documents VLM (données internes).
              </p>
              <div className="flex gap-2 flex-wrap justify-center mt-2">
                {[
                  "Quelles sont les tendances du marché WAAM en 2025 ?",
                  "Quels produits VLM supportent le titane ?",
                  "Bonnes pratiques WAAM aérospatiale et produits VLM associés ?",
                ].map((s) => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
                    className="text-xs border rounded-full px-3 py-1.5 hover:bg-muted transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) =>
            msg.role === "user" ? (
              <UserBubble key={msg.id} content={msg.content} />
            ) : (
              <AssistantMessage
                key={msg.id}
                content={msg.content}
                sources={msg.sources}
                isStreaming={msg.isStreaming}
              >
                {msg.currentStep && <StepBadge step={msg.currentStep} />}
                {msg.agent && !msg.currentStep && <AgentBadge agent={msg.agent} />}
              </AssistantMessage>
            )
          )}
          {isLoading && messages[messages.length - 1]?.role !== "assistant" && <LoadingDots />}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div className="shrink-0 border-t bg-background px-4 py-3">
        <div className="flex items-end gap-2 max-w-[680px] mx-auto">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Posez votre question…"
            rows={1}
            disabled={isLoading}
            className="flex-1 resize-none rounded-xl border bg-background px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50 min-h-[44px] max-h-[200px]"
            style={{ height: "auto" }}
            onInput={(e) => {
              const el = e.currentTarget;
              el.style.height = "auto";
              el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
            }}
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading}
            className="shrink-0 rounded-xl bg-primary text-primary-foreground p-3 hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
        <p className="text-center text-xs text-muted-foreground mt-2">
          Données VLM sensibles protégées · Requêtes externes sans données internes
        </p>
      </div>
    </div>
  );
}
