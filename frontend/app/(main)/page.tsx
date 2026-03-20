"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowDown, Brain, Send } from "lucide-react";
import { fetchCollections, fetchLLMStatus, getConversationMessages, streamChat } from "@/app/lib/api";
import type { ChatMessage } from "@/app/lib/types";
import { useConversation } from "@/app/providers";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { Spinner } from "@/components/ui/spinner";
import { AssistantMessage, LoadingDots, UserBubble } from "@/components/chat/MarkdownMessage";

// ─── Suggestion Cards ──────────────────────────────────────────────────────────
const SUGGESTIONS = [
  { title: "Résume un document", subtitle: "Explique les points clés d'un fichier" },
  { title: "Cherche une information", subtitle: "Pose une question sur tes données" },
  { title: "Compare des éléments", subtitle: "Analyse les différences entre documents" },
  { title: "Aide à la rédaction", subtitle: "Génère du contenu à partir de tes sources" },
];

// ─── Message item ──────────────────────────────────────────────────────────────
function MessageItem({ msg }: { msg: ChatMessage }) {
  if (msg.role === "user") return <UserBubble content={msg.content} />;
  return (
    <AssistantMessage
      content={msg.content}
      sources={msg.sources}
      metrics={msg.metrics}
      isStreaming={msg.isStreaming}
    />
  );
}


// ─── Reasoning progress banner ─────────────────────────────────────────────────
const REASONING_STEP_LABELS: Record<string, string> = {
  exploring:  "Exploration large des documents…",
  reflecting: "Analyse des résultats, identification des lacunes…",
  deepening:  "Approfondissement ciblé…",
  generating: "Génération de la réponse synthétisée…",
};

function ReasoningBanner({
  step,
}: {
  step: { step: string; query?: string; index?: number; total?: number; found?: string; gaps_count?: number };
}) {
  const label = REASONING_STEP_LABELS[step.step] ?? step.step;
  const deepenDetail = step.step === "deepening" && step.query
    ? `[${step.index}/${step.total}] ${step.query}`
    : null;
  const reflectDetail = step.step === "reflecting" && step.found
    ? step.found
    : null;

  return (
    <div className="flex gap-3 mb-4">
      <div className="w-8 shrink-0" />
      <div className="flex-1 rounded-lg border border-violet-500/20 bg-violet-500/5 px-4 py-3 max-w-[580px]">
        <div className="flex items-center gap-2 mb-1">
          <Spinner className="h-3.5 w-3.5 text-violet-500" />
          <span className="text-xs font-semibold text-violet-600 dark:text-violet-400">
            Mode Raisonnement
          </span>
          {step.step === "deepening" && step.total != null && (
            <span className="ml-auto text-xs bg-violet-500/15 rounded-full px-2 py-0.5 text-violet-700 dark:text-violet-300 shrink-0">
              {step.index}/{step.total}
            </span>
          )}
          {step.step === "reflecting" && step.gaps_count != null && (
            <span className="ml-auto text-xs bg-violet-500/15 rounded-full px-2 py-0.5 text-violet-700 dark:text-violet-300 shrink-0">
              {step.gaps_count} lacune{step.gaps_count !== 1 ? "s" : ""}
            </span>
          )}
        </div>
        <p className="text-xs text-muted-foreground">{label}</p>
        {deepenDetail && (
          <p className="text-[11px] text-muted-foreground/70 truncate mt-0.5">{deepenDetail}</p>
        )}
        {reflectDetail && (
          <p className="text-[11px] text-muted-foreground/70 line-clamp-2 mt-0.5">{reflectDetail}</p>
        )}
      </div>
    </div>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────────
export default function ChatPage() {
  const { currentConversationId, createConversation, refreshConversations, mode } =
    useConversation();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [llmReady, setLlmReady] = useState(false);
  const [collections, setCollections] = useState<string[]>([]);
  const [collection, setCollection] = useState("");
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reasoningMode, setReasoningMode] = useState(false);
  const [reasoningStep, setReasoningStep] = useState<{ step: string; query?: string; index?: number; total?: number; found?: string; gaps_count?: number } | null>(null);

  const messagesRef = useRef<ChatMessage[]>(messages);
  messagesRef.current = messages;

  const activeConvIdRef = useRef<string | null>(null);
  const skipNextReloadRef = useRef(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // ── LLM status polling ────────────────────────────────────────────────────
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    let attempts = 0;
    const check = async () => {
      attempts++;
      const ready = await fetchLLMStatus();
      if (ready) {
        setLlmReady(true);
        clearInterval(interval);
      } else if (attempts >= 100) {
        setError("Le modèle IA n'a pas pu démarrer. Vérifiez qu'Ollama est lancé.");
        clearInterval(interval);
      }
    };
    check();
    interval = setInterval(check, 3000);
    return () => clearInterval(interval);
  }, []);

  // ── Fetch collections ───────────────────────────────────────────────────────
  useEffect(() => {
    fetchCollections().then((cols) => {
      setCollections(cols);
      if (cols.length > 0) setCollection(cols[0]);
    });
  }, []);

  // ── Load messages when conversation changes (sidebar click) ─────────────────
  useEffect(() => {
    if (skipNextReloadRef.current) {
      skipNextReloadRef.current = false;
      return;
    }
    if (currentConversationId) {
      activeConvIdRef.current = currentConversationId;
      getConversationMessages(currentConversationId).then((msgs) => {
        setMessages(
          msgs.map((m) => ({
            id: m.id,
            role: m.role as "user" | "assistant",
            content: m.content,
          }))
        );
      });
    } else {
      activeConvIdRef.current = null;
      setMessages([]);
    }
  }, [currentConversationId]);

  // ── Auto-scroll on new messages ─────────────────────────────────────────────
  const scrollToBottom = useCallback((smooth = true) => {
    messagesEndRef.current?.scrollIntoView({ behavior: smooth ? "smooth" : "instant" });
  }, []);

  useEffect(() => {
    if (messages.length > 0) scrollToBottom(false);
  }, [messages.length, scrollToBottom]);

  // ── Auto-resize textarea ────────────────────────────────────────────────────
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`; // max ~6 lines
  }, [input]);

  // ── Scroll detection ────────────────────────────────────────────────────────
  const handleScroll = () => {
    const el = scrollAreaRef.current;
    if (!el) return;
    setShowScrollBtn(el.scrollHeight - el.scrollTop - el.clientHeight > 100);
  };

  // ── Send message ────────────────────────────────────────────────────────────
  const handleSend = useCallback(
    async (content?: string) => {
      const text = (content ?? input).trim();
      if (!text || isLoading || !llmReady || !collection) return;

      setInput("");
      setError(null);
      setReasoningStep(null);

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
      };
      const assistantId = crypto.randomUUID();

      const currentMessages = messagesRef.current;
      const history = currentMessages
        .filter((m) => m.content && !m.isStreaming)
        .slice(-10)
        .map((m) => ({ role: m.role, content: m.content }));

      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      scrollToBottom();

      // Auto-create conversation on first message
      let convId = activeConvIdRef.current;
      const isFirstMessage = currentMessages.length === 0;
      if (isFirstMessage) {
        skipNextReloadRef.current = true;
        convId = await createConversation(text.slice(0, 50), mode);
        activeConvIdRef.current = convId;
      }

      let assistantContent = "";

      try {
        let firstToken = true;
        let doneReceived = false;

        for await (const event of streamChat(text, collection, "defaut", history, convId ?? undefined, reasoningMode)) {
          if (event.error) {
            setError(event.error);
            setIsLoading(false);
            break;
          }

          if (event.reasoning_step) {
            setReasoningStep(event.reasoning_step);
          }

          if (event.token) {
            setReasoningStep(null); // clear banner when first token arrives
            assistantContent += event.token;

            if (firstToken) {
              firstToken = false;
              setIsLoading(false);
              setMessages((prev) => [
                ...prev,
                { id: assistantId, role: "assistant", content: assistantContent, isStreaming: true },
              ]);
            } else {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, content: assistantContent } : m
                )
              );
            }
            scrollToBottom();
          }

          if (event.done) {
            doneReceived = true;
            setReasoningStep(null);
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, sources: event.sources ?? [], metrics: event.metrics, isStreaming: false }
                  : m
              )
            );
            break;
          }
        }

        // Connexion interrompue sans event done
        if (!doneReceived) {
          setReasoningStep(null);
          if (assistantContent === "") {
            // Aucun token reçu — spinner fantôme
            setIsLoading(false);
            setError("La connexion au serveur a été interrompue. Veuillez réessayer.");
          } else {
            // Réponse partielle reçue — finaliser le message
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, isStreaming: false } : m
              )
            );
            setError("Réponse peut-être incomplète (connexion interrompue).");
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Erreur de connexion";
        setError(msg);
        setIsLoading(false);
        setReasoningStep(null);
      } finally {
        // Le backend persiste les messages directement (résistant aux déconnexions SSE)
        if (convId) refreshConversations();
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [input, isLoading, llmReady, collection, mode, createConversation, refreshConversations, scrollToBottom, reasoningMode]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isWelcome = messages.length === 0 && !isLoading;

  // Shared input bar JSX (rendered in two different positions depending on state)
  const inputBar = (
    <>
      <div className="flex items-end gap-3 bg-card border border-border rounded-2xl px-4 py-3 shadow-sm">
        {/* Reasoning mode toggle */}
        <button
          type="button"
          onClick={() => setReasoningMode((v) => !v)}
          title={reasoningMode ? "Mode Raisonnement actif — cliquer pour désactiver" : "Activer le Mode Raisonnement (multi-requêtes RAG)"}
          className={cn(
            "flex-shrink-0 flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-xs border transition-all",
            reasoningMode
              ? "bg-violet-500/15 border-violet-500/30 text-violet-600 dark:text-violet-400"
              : "bg-transparent border-border text-muted-foreground hover:text-foreground"
          )}
        >
          <Brain size={12} />
          <span className="hidden sm:inline">Raisonnement</span>
        </button>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message..."
          rows={1}
          disabled={!llmReady || !!error}
          className="flex-1 bg-transparent text-fluid-sm text-foreground resize-none outline-none placeholder:text-muted-foreground leading-relaxed overflow-hidden disabled:cursor-not-allowed"
          style={{ minHeight: "24px" }}
        />
        <button
          onClick={() => handleSend()}
          disabled={!input.trim() || isLoading || !llmReady || !collection}
          className={cn(
            "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all",
            input.trim() && !isLoading && llmReady && collection
              ? "bg-foreground text-background hover:opacity-80"
              : "bg-muted text-muted-foreground cursor-not-allowed"
          )}
        >
          <Send size={14} />
        </button>
      </div>
      {/* Collection selector + disclaimer */}
      <div className="flex items-center justify-between mt-2 px-1">
        <select
          value={collection}
          onChange={(e) => setCollection(e.target.value)}
          className="text-fluid-xs text-muted-foreground bg-transparent border-none outline-none cursor-pointer hover:text-foreground transition-colors"
        >
          {collections.length === 0 ? (
            <option value="">Aucune collection</option>
          ) : (
            collections.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))
          )}
        </select>
        <p className="text-fluid-xs text-muted-foreground/60">
          Ce système peut faire des erreurs.
        </p>
      </div>
    </>
  );

  return (
    <div className="relative flex flex-col h-full bg-background">
      {/* ── Minimal header ───────────────────────────────────────────────────── */}
      <header className="flex items-center h-14 px-4 border-b border-border flex-shrink-0">
        <SidebarTrigger className="text-muted-foreground hover:text-foreground" />
      </header>

      {/* ── LLM loading banner ──────────────────────────────────────────────── */}
      {!llmReady && !error && (
        <div className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 text-sm text-amber-600 dark:text-amber-400">
          <Spinner className="h-3.5 w-3.5" />
          Modèle IA en cours de chargement… (peut prendre 1–2 min au démarrage)
        </div>
      )}

      {isWelcome ? (
        /* ── WELCOME STATE : input centré ────────────────────────────────── */
        <div className="flex-1 overflow-y-auto flex flex-col items-center px-4 pt-[10vh] md:pt-[14vh]">
          <h1 className="text-fluid-2xl font-semibold text-foreground mb-8">
            Comment puis-je vous aider ?
          </h1>

          {/* Suggestion cards */}
          <div className="grid grid-cols-2 gap-3 w-full max-w-[720px] mb-5">
            {SUGGESTIONS.map((s, i) => (
              <button
                key={i}
                onClick={() => handleSend(s.title)}
                className="text-left p-4 rounded-xl bg-card border border-border hover:border-foreground/20 hover:bg-muted/60 transition-all"
              >
                <p className="text-fluid-sm font-medium text-foreground mb-1">{s.title}</p>
                <p className="text-fluid-xs text-muted-foreground">{s.subtitle}</p>
              </button>
            ))}
          </div>

          {/* Input bar centered */}
          <div className="w-full max-w-[720px]">
            {inputBar}
          </div>
        </div>
      ) : (
        /* ── CHAT STATE : messages + input en bas ────────────────────────── */
        <>
          <div
            ref={scrollAreaRef}
            onScroll={handleScroll}
            className="flex-1 overflow-y-auto"
          >
            <div className="max-w-[720px] w-full mx-auto px-4 md:px-6 py-6 md:py-10">
              {messages.map((msg) => (
                <MessageItem key={msg.id} msg={msg} />
              ))}
              {isLoading && reasoningStep && (
                <ReasoningBanner step={reasoningStep} />
              )}
              {isLoading && !reasoningStep && <LoadingDots />}
              {error && (
                <p className="text-center text-sm text-destructive py-2">{error}</p>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Scroll-to-bottom button */}
          {showScrollBtn && (
            <button
              onClick={() => scrollToBottom()}
              className="absolute bottom-28 right-6 w-9 h-9 rounded-full bg-card border border-border shadow-md flex items-center justify-center text-muted-foreground hover:text-foreground transition-all"
            >
              <ArrowDown size={16} />
            </button>
          )}

          {/* Input bar fixed at bottom */}
          <div className="flex-shrink-0 px-4 pb-4 pt-2 bg-background">
            <div className="max-w-[720px] mx-auto">
              {inputBar}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
