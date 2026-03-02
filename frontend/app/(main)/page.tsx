"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { ArrowDown, Check, Copy, Send } from "lucide-react";
import Image from "next/image";
import { addMessage, fetchCollections, fetchLLMStatus, getConversationMessages, streamChat } from "@/app/lib/api";
import type { ChatMessage, ChatSource } from "@/app/lib/types";
import { useConversation } from "@/app/providers";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";

// ─── Suggestion Cards ──────────────────────────────────────────────────────────
const SUGGESTIONS = [
  { title: "Résume un document", subtitle: "Explique les points clés d'un fichier" },
  { title: "Cherche une information", subtitle: "Pose une question sur tes données" },
  { title: "Compare des éléments", subtitle: "Analyse les différences entre documents" },
  { title: "Aide à la rédaction", subtitle: "Génère du contenu à partir de tes sources" },
];

// ─── Inline / Block Code ───────────────────────────────────────────────────────
function CodeBlock({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLElement>) {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || "");
  const language = match?.[1] ?? "";
  const code = String(children).replace(/\n$/, "");

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!match) {
    return (
      <code
        className="bg-muted text-orange-400 dark:text-orange-300 px-1.5 py-0.5 rounded text-[0.85em] font-mono"
        {...props}
      >
        {children}
      </code>
    );
  }

  return (
    <div className="relative my-4 rounded-lg overflow-hidden border border-border">
      <div className="flex items-center justify-between px-4 py-2 bg-muted/60 border-b border-border">
        <span className="text-fluid-xs text-muted-foreground font-mono">{language}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 text-fluid-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? "Copié !" : "Copier"}
        </button>
      </div>
      <pre className="overflow-x-auto p-4 text-fluid-sm m-0 bg-[#0d1117]">
        <code className={className} {...props}>
          {children}
        </code>
      </pre>
    </div>
  );
}

// ─── Message item ──────────────────────────────────────────────────────────────
function MessageItem({ msg }: { msg: ChatMessage }) {
  if (msg.role === "user") {
    return (
      <div className="flex justify-end mb-6">
        <div className="max-w-[80%] bg-muted text-foreground rounded-[18px] px-4 py-3 text-fluid-sm leading-relaxed whitespace-pre-wrap">
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 mb-6">
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center mt-0.5 border border-border">
        <Image
          src="/logoVLM.png"
          alt="RAG Local"
          width={18}
          height={18}
          className="object-contain"
        />
      </div>
      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="prose prose-base dark:prose-invert max-w-none text-foreground leading-relaxed">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{
              code: CodeBlock as React.ComponentType<React.HTMLAttributes<HTMLElement>>,
              a: ({ href, children }) => (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-500 hover:underline"
                >
                  {children}
                </a>
              ),
            }}
          >
            {msg.content || (msg.isStreaming ? "\u200b" : "")}
          </ReactMarkdown>
          {msg.isStreaming && (
            <span className="inline-block w-[2px] h-4 bg-foreground/60 animate-pulse align-middle ml-0.5" />
          )}
        </div>
        {/* Sources */}
        {msg.sources && msg.sources.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {msg.sources.map((src: ChatSource, i: number) => (
              <span
                key={i}
                className="text-fluid-xs text-muted-foreground bg-muted border border-border rounded px-2 py-1"
              >
                {src.fichier} · p.{src.page}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Loading dots ──────────────────────────────────────────────────────────────
function LoadingDots() {
  return (
    <div className="flex gap-3 mb-6">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center border border-border">
        <Image src="/logoVLM.png" alt="RAG Local" width={18} height={18} className="object-contain" />
      </div>
      <div className="flex items-center gap-1.5 mt-2">
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:-0.3s]" />
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:-0.15s]" />
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce" />
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
      if (!text || isLoading || !llmReady) return;

      setInput("");
      setError(null);

      const userMsg: ChatMessage = {
        id: Math.random().toString(36).slice(2),
        role: "user",
        content: text,
      };
      const assistantId = Math.random().toString(36).slice(2);

      const history = messages
        .filter((m) => m.content && !m.isStreaming)
        .slice(-10)
        .map((m) => ({ role: m.role, content: m.content }));

      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      scrollToBottom();

      // Auto-create conversation on first message
      let convId = activeConvIdRef.current;
      const isFirstMessage = messages.length === 0;
      if (isFirstMessage) {
        skipNextReloadRef.current = true;
        convId = await createConversation(text.slice(0, 50), mode);
        activeConvIdRef.current = convId;
      }

      let assistantContent = "";

      try {
        let firstToken = true;

        for await (const event of streamChat(text, collection, "defaut", history)) {
          if (event.error) {
            setError(event.error);
            setIsLoading(false);
            break;
          }

          if (event.token) {
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
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, sources: event.sources ?? [], isStreaming: false }
                  : m
              )
            );
            break;
          }
        }

        // Edge case: no tokens received
        if (assistantContent === "") setIsLoading(false);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Erreur de connexion";
        setError(msg);
        setIsLoading(false);
      } finally {
        // Persist to DB
        if (convId) {
          await addMessage(convId, "user", text);
          if (assistantContent) await addMessage(convId, "assistant", assistantContent);
          refreshConversations();
        }
      }
    },
    [input, isLoading, llmReady, messages, collection, mode, createConversation, refreshConversations, scrollToBottom]
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
          disabled={!input.trim() || isLoading || !llmReady}
          className={cn(
            "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all",
            input.trim() && !isLoading && llmReady
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
          <svg className="animate-spin h-3.5 w-3.5 shrink-0" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
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
              {isLoading && <LoadingDots />}
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
