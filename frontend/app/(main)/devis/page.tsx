"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { ArrowDown, ChevronDown, Download, FileText, Send, ShoppingCart, Trash2, X } from "lucide-react";
import Image from "next/image";
import {
  addElementToPanierDirect,
  addMessage,
  clearPanier,
  exportDevisExcel,
  fetchCatalogStatus,
  fetchCollections,
  fetchLLMStatus,
  fetchPanier,
  fetchPosteElements,
  getConversationMessages,
  lockDevisAffaire,
  removePanierItem,
  setSearchScope,
  streamDevisChat,
  streamGenerateDevis,
} from "@/app/lib/api";
import type { CatalogElement, ChatMessage, PanierItem } from "@/app/lib/types";
import { useConversation } from "@/app/providers";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";

// ─── Tool call status badge ────────────────────────────────────────────────────
const TOOL_LABELS: Record<string, string> = {
  search_catalog: "Recherche dans le catalogue",
  search_docs: "Recherche dans les documents",
  add_to_panier: "Ajout au panier",
  ask_user_choice: "Présentation des options",
};

interface ToolCallState {
  id: string;
  name: string;
  status: "running" | "done";
}

function ToolCallBadge({ tool }: { tool: ToolCallState }) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 text-xs rounded-full px-3 py-1.5 w-fit mb-2 border",
        tool.status === "running"
          ? "bg-amber-500/10 border-amber-500/20 text-amber-600 dark:text-amber-400"
          : "bg-muted border-border text-muted-foreground"
      )}
    >
      {tool.status === "running" ? (
        <svg className="animate-spin h-3 w-3 shrink-0" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
        </svg>
      ) : (
        <span className="text-green-500">✓</span>
      )}
      {TOOL_LABELS[tool.name] ?? tool.name}
    </div>
  );
}

// ─── Query variant selector ────────────────────────────────────────────────────
type VariantOption = {
  id: string;
  label: string;
  detail?: string;
  description?: string;
  columns?: string[];
  rows?: (string | number | null)[][];
};

function QueryVariantSelector({
  question,
  options,
  onSelect,
  disabled,
}: {
  question: string;
  options: VariantOption[];
  onSelect: (id: string, label: string) => void;
  disabled?: boolean;
}) {
  const [activeIdx, setActiveIdx] = useState(0);
  const active = options[activeIdx];

  return (
    <div className="max-w-2xl">
      <p className="text-sm text-foreground mb-3">{question}</p>

      {/* Horizontal cards */}
      <div className="flex gap-2 overflow-x-auto pb-2 mb-3">
        {options.map((opt, i) => (
          <button
            key={opt.id}
            disabled={disabled}
            onClick={() => setActiveIdx(i)}
            className={cn(
              "flex-shrink-0 rounded-lg border p-3 text-left transition-all min-w-[160px] max-w-[220px]",
              i === activeIdx
                ? "border-foreground/50 bg-muted shadow-sm"
                : "border-border bg-card hover:bg-muted/40 hover:border-foreground/20",
              disabled && "opacity-50 cursor-not-allowed"
            )}
          >
            <p className="text-xs font-medium text-foreground truncate">{opt.label}</p>
            {opt.description && (
              <span className="inline-block mt-1 text-xs bg-foreground/10 text-foreground/70 rounded-full px-2 py-0.5">
                {opt.description}
              </span>
            )}
            {opt.detail && (
              <p className="text-xs text-muted-foreground mt-1 truncate">{opt.detail}</p>
            )}
          </button>
        ))}
      </div>

      {/* Table for active card */}
      {active?.columns && active?.rows && active.rows.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-border mb-3 bg-card">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border bg-muted/50">
                {active.columns.map((col) => (
                  <th key={col} className="px-3 py-2 text-left text-muted-foreground font-medium whitespace-nowrap">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {active.rows.map((row, ri) => (
                <tr key={ri} className={cn("border-b border-border/40 last:border-0", ri % 2 !== 0 && "bg-muted/20")}>
                  {row.map((cell, ci) => (
                    <td key={ci} className="px-3 py-2 text-foreground">{cell ?? "—"}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Confirm */}
      {!disabled && (
        <button
          onClick={() => onSelect(active.id, active.label)}
          className="rounded-full bg-foreground text-background px-4 py-1.5 text-sm font-medium hover:opacity-80 transition-all"
        >
          Sélectionner « {active?.label} »
        </button>
      )}
    </div>
  );
}

// ─── Message item ──────────────────────────────────────────────────────────────
function MessageItem({
  msg,
  toolCalls,
  onChoiceSelect,
}: {
  msg: ChatMessage;
  toolCalls?: ToolCallState[];
  onChoiceSelect?: (id: string, label: string, messageId: string) => void | Promise<void>;
}) {
  if (msg.role === "user") {
    return (
      <div className="flex justify-end mb-6">
        <div className="max-w-[80%] bg-muted text-foreground rounded-[18px] px-4 py-3 text-fluid-sm leading-relaxed whitespace-pre-wrap">
          {msg.content}
        </div>
      </div>
    );
  }

  // Choice cards
  if (msg.type === "choices" && msg.choices) {
    const choiceType = msg.choices.type;

    // Rich variant selector for poste / affaire (have table data)
    if (choiceType === "poste" || choiceType === "affaire") {
      return (
        <div className="flex gap-3 mb-6">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center mt-0.5 border border-border">
            <Image src="/logoVLM.png" alt="Devis" width={18} height={18} className="object-contain" />
          </div>
          <QueryVariantSelector
            question={msg.choices.question}
            options={msg.choices.options}
            disabled={!!msg.choiceSelected}
            onSelect={(id, label) => onChoiceSelect?.(id, label, msg.id)}
          />
        </div>
      );
    }

    // Simple buttons for yes/no choices (search_scope, relevance, element, etc.)
    return (
      <div className="flex gap-3 mb-6">
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center mt-0.5 border border-border">
          <Image src="/logoVLM.png" alt="Devis" width={18} height={18} className="object-contain" />
        </div>
        <div className="max-w-xl">
          <p className="text-sm text-foreground mb-3">{msg.choices.question}</p>
          <div className="flex flex-wrap gap-2">
            {msg.choices.options.map((opt) => (
              <button
                key={opt.id}
                disabled={!!msg.choiceSelected}
                onClick={() => onChoiceSelect?.(opt.id, opt.label, msg.id)}
                title={opt.detail}
                className="rounded-full border border-border bg-card px-3 py-1.5 text-sm hover:bg-muted/60 hover:border-foreground/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 mb-6">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center mt-0.5 border border-border">
        <Image src="/logoVLM.png" alt="Devis" width={18} height={18} className="object-contain" />
      </div>
      <div className="flex-1 min-w-0">
        {/* Tool call indicators attached to this message */}
        {toolCalls && toolCalls.map((tc) => <ToolCallBadge key={tc.id} tool={tc} />)}
        <div className="prose prose-base dark:prose-invert max-w-none text-foreground leading-relaxed">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{
              a: ({ href, children }) => (
                <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
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
      </div>
    </div>
  );
}

// ─── Loading dots ──────────────────────────────────────────────────────────────
function LoadingDots() {
  return (
    <div className="flex gap-3 mb-6">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center border border-border">
        <Image src="/logoVLM.png" alt="Devis" width={18} height={18} className="object-contain" />
      </div>
      <div className="flex items-center gap-1.5 mt-2">
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:-0.3s]" />
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:-0.15s]" />
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce" />
      </div>
    </div>
  );
}

// ─── Panier panel ─────────────────────────────────────────────────────────────
function PanierPanel({
  panier,
  onRemoveItem,
  onClear,
  onGenerate,
  onExportExcel,
  isGenerating,
  isExporting,
}: {
  panier: PanierItem[];
  onRemoveItem: (id: string) => void;
  onClear: () => void;
  onGenerate: () => void;
  onExportExcel: () => void;
  isGenerating: boolean;
  isExporting: boolean;
}) {
  // null = loading, [] = vide, [...] = chargé
  const [expandedItems, setExpandedItems] = useState<Record<string, CatalogElement[] | null>>({});

  const toggleExpand = async (item: PanierItem) => {
    if (expandedItems[item.id] !== undefined) {
      setExpandedItems((prev) => {
        const n = { ...prev };
        delete n[item.id];
        return n;
      });
    } else {
      setExpandedItems((prev) => ({ ...prev, [item.id]: null }));
      const elems = await fetchPosteElements(item.nom_poste, item.nom_affaire ?? undefined);
      setExpandedItems((prev) => ({ ...prev, [item.id]: elems }));
    }
  };

  return (
    <aside className="w-72 flex-shrink-0 border-l border-border flex flex-col bg-card h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <ShoppingCart className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Panier ({panier.length})</span>
        </div>
        <button
          onClick={onClear}
          title="Vider le panier"
          className="text-muted-foreground hover:text-destructive transition-colors"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Items */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-2">
        {panier.map((item) => {
          const expanded = expandedItems[item.id] !== undefined;
          const elements = expandedItems[item.id];
          return (
            <div
              key={item.id}
              className="group rounded-lg border border-border bg-background p-2.5"
            >
              <div className="flex items-start gap-2">
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate">{item.nom_poste}</p>
                  {item.ensemble && (
                    <p className="text-xs text-muted-foreground truncate">{item.ensemble}</p>
                  )}
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-muted-foreground">Qté {item.quantite}</span>
                    {item.fournisseur && (
                      <span className="text-xs text-muted-foreground truncate">· {item.fournisseur}</span>
                    )}
                  </div>
                  {item.nom_affaire && (
                    <p className="text-xs text-muted-foreground/60 truncate mt-0.5">Réf. {item.nom_affaire}</p>
                  )}
                </div>
                <div className="flex items-center gap-1 shrink-0 mt-0.5">
                  <button
                    onClick={() => toggleExpand(item)}
                    title="Voir les éléments"
                    className="text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <ChevronDown className={cn("h-3 w-3 transition-transform", expanded && "rotate-180")} />
                  </button>
                  <button
                    onClick={() => onRemoveItem(item.id)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                    title="Retirer"
                  >
                    <X className="h-3 w-3 text-muted-foreground hover:text-destructive" />
                  </button>
                </div>
              </div>

              {/* Accordion: sub-elements */}
              {expanded && (
                <div className="mt-2 pl-2 border-l border-border space-y-1">
                  {elements === null ? (
                    <p className="text-xs text-muted-foreground">Chargement…</p>
                  ) : elements.length === 0 ? (
                    <p className="text-xs text-muted-foreground">Aucun détail disponible</p>
                  ) : (
                    elements.map((el, i) => (
                      <div key={i} className="text-xs text-muted-foreground">
                        <span className="font-medium">{el.elements || "—"}</span>
                        {el.fournisseur && <span> · {el.fournisseur}</span>}
                        {el.fourniture != null && <span> · {el.fourniture}€</span>}
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="px-3 py-3 border-t border-border space-y-2">
        <button
          onClick={onGenerate}
          disabled={isGenerating || panier.length === 0}
          className={cn(
            "w-full flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all",
            isGenerating || panier.length === 0
              ? "bg-muted text-muted-foreground cursor-not-allowed"
              : "bg-foreground text-background hover:opacity-80"
          )}
        >
          <FileText className="h-4 w-4" />
          {isGenerating ? "Génération…" : "Générer le devis"}
        </button>
        <button
          onClick={onExportExcel}
          disabled={isExporting || panier.length === 0}
          className={cn(
            "w-full flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all border",
            isExporting || panier.length === 0
              ? "border-border bg-muted text-muted-foreground cursor-not-allowed"
              : "border-green-600/40 bg-green-600/10 text-green-700 dark:text-green-400 hover:bg-green-600/20"
          )}
        >
          <Download className="h-4 w-4" />
          {isExporting ? "Export…" : "Télécharger Excel"}
        </button>
      </div>
    </aside>
  );
}

// ─── Welcome screen ────────────────────────────────────────────────────────────
const SUGGESTIONS = [
  {
    title: "Décrire les specs techniques",
    subtitle: "Je recherche les composants compatibles dans les documents",
  },
  {
    title: "Référencer une affaire passée",
    subtitle: "Je retrouve les postes d'un projet similaire dans le catalogue",
  },
  {
    title: "Chercher un composant",
    subtitle: "Recherche par nom de poste, fournisseur ou ensemble",
  },
  {
    title: "Affiner un devis existant",
    subtitle: "Modifier quantités, alternatives ou compléments",
  },
];

// ─── Main page ─────────────────────────────────────────────────────────────────
export default function DevisPage() {
  const { currentConversationId, createConversation, refreshConversations, mode } = useConversation();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [llmReady, setLlmReady] = useState(false);
  const [collections, setCollections] = useState<string[]>([]);
  const [collection, setCollection] = useState("");
  const [catalogLoaded, setCatalogLoaded] = useState(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Tool calls for the current streaming turn
  const [activeToolCalls, setActiveToolCalls] = useState<ToolCallState[]>([]);
  // Map of assistantMessageId → tool calls that belong to it
  const [messageToolCalls, setMessageToolCalls] = useState<Record<string, ToolCallState[]>>({});

  const [panier, setPanier] = useState<PanierItem[]>([]);

  const activeConvIdRef = useRef<string | null>(null);
  const skipNextReloadRef = useRef<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scopeChoicePendingRef = useRef(false);
  const pendingMessageRef = useRef<string>("");

  // ── LLM status + catalog status ──────────────────────────────────────────
  useEffect(() => {
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
    const interval = setInterval(check, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetchCatalogStatus().then((s) => setCatalogLoaded(s.loaded));
    fetchCollections().then((cols) => {
      setCollections(cols);
      if (cols.length > 0) setCollection(cols[0]);
    });
  }, []);

  // ── Load conversation messages when switching conversation ────────────────
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
      fetchPanier(currentConversationId).then(setPanier);
    } else {
      activeConvIdRef.current = null;
      setMessages([]);
      setPanier([]);
    }
  }, [currentConversationId]);

  // ── Auto-scroll ───────────────────────────────────────────────────────────
  const scrollToBottom = useCallback((smooth = true) => {
    messagesEndRef.current?.scrollIntoView({ behavior: smooth ? "smooth" : "instant" });
  }, []);

  useEffect(() => {
    if (messages.length > 0) scrollToBottom(false);
  }, [messages.length, scrollToBottom]);

  // ── Auto-resize textarea ─────────────────────────────────────────────────
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`;
  }, [input]);

  const handleScroll = () => {
    const el = scrollAreaRef.current;
    if (!el) return;
    setShowScrollBtn(el.scrollHeight - el.scrollTop - el.clientHeight > 100);
  };

  // ── Send message ──────────────────────────────────────────────────────────
  const handleSend = useCallback(
    async (content?: string) => {
      const text = (content ?? input).trim();
      if (!text || isLoading || !llmReady) return;

      // Intercept: if a scope choice is pending, show choice cards first
      if (scopeChoicePendingRef.current) {
        pendingMessageRef.current = text;
        setInput("");
        const choiceId = Math.random().toString(36).slice(2);
        const question = "Souhaitez-vous continuer dans la même affaire ou chercher dans tout le catalogue ?";
        setMessages((prev) => [
          ...prev,
          {
            id: choiceId,
            role: "assistant" as const,
            content: question,
            type: "choices" as const,
            choices: {
              type: "search_scope" as const,
              question,
              options: [
                { id: JSON.stringify({ action: "search_same_affaire" }), label: "Même affaire" },
                { id: JSON.stringify({ action: "search_all_affaires" }), label: "Tout le catalogue" },
              ],
            },
            choiceSelected: false,
          },
        ]);
        return;
      }

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
      setActiveToolCalls([]);
      scrollToBottom();

      // Auto-create conversation on first message
      let convId = activeConvIdRef.current;
      if (messages.length === 0) {
        skipNextReloadRef.current = true;
        convId = await createConversation(text.slice(0, 50), mode);
        activeConvIdRef.current = convId;
      }

      let assistantContent = "";
      const turnToolCalls: ToolCallState[] = [];

      try {
        let firstToken = true;

        for await (const event of streamDevisChat(text, collection, convId!, history)) {
          if (event.error) {
            setError(event.error);
            setIsLoading(false);
            break;
          }

          // Docs search result — show which sources were consulted
          if (event.docs_result) {
            const { query, sources, count } = event.docs_result;
            const sourceList = sources.length > 0 ? sources.join(", ") : "aucune source";
            const infoId = Math.random().toString(36).slice(2);
            const infoContent = count > 0
              ? `📄 Documentation consultée pour « ${query} » — ${count} passage(s) trouvé(s) dans : ${sourceList}`
              : `📄 Aucun résultat dans la documentation pour « ${query} »`;
            setMessages((prev) => [
              ...prev,
              { id: infoId, role: "assistant" as const, content: infoContent, isStreaming: false },
            ]);
          }

          // Tool call status update
          if (event.tool_call) {
            const { name, status } = event.tool_call;
            const tcId = `${name}-${Date.now()}`;

            if (status === "running") {
              const newTc: ToolCallState = { id: tcId, name, status: "running" };
              turnToolCalls.push(newTc);
              setActiveToolCalls([...turnToolCalls]);
            } else {
              // Mark the matching running tool as done
              const idx = turnToolCalls.findLastIndex(
                (tc) => tc.name === name && tc.status === "running"
              );
              if (idx !== -1) turnToolCalls[idx] = { ...turnToolCalls[idx], status: "done" };
              setActiveToolCalls([...turnToolCalls]);
            }
          }

          // Panier update
          if (event.panier) {
            setPanier(event.panier);
          }

          // Choice cards — LLM wants user to pick one option
          if (event.choices) {
            const choiceQuestion = event.choices.question;
            assistantContent = choiceQuestion;
            setIsLoading(false);
            setActiveToolCalls([]);
            setMessageToolCalls((prev) => ({ ...prev, [assistantId]: [...turnToolCalls] }));
            setMessages((prev) => [
              ...prev,
              {
                id: assistantId,
                role: "assistant",
                content: choiceQuestion,
                type: "choices",
                choices: event.choices,
                choiceSelected: false,
                isStreaming: false,
              },
            ]);
            firstToken = false;
          }

          // Streaming token
          if (event.token) {
            assistantContent += event.token;

            if (firstToken) {
              firstToken = false;
              setIsLoading(false);
              // Snapshot tool calls into this message
              setMessageToolCalls((prev) => ({ ...prev, [assistantId]: [...turnToolCalls] }));
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
            if (event.panier) setPanier(event.panier);
            if (event.ask_scope) scopeChoicePendingRef.current = true;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, isStreaming: false } : m
              )
            );
            setActiveToolCalls([]);
            break;
          }
        }

        if (assistantContent === "") setIsLoading(false);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Erreur de connexion";
        setError(msg);
        setIsLoading(false);
      } finally {
        if (convId) {
          await addMessage(convId, "user", text);
          if (assistantContent) await addMessage(convId, "assistant", assistantContent);
          refreshConversations();
        }
      }
    },
    [input, isLoading, llmReady, messages, collection, mode, createConversation, refreshConversations, scrollToBottom]
  );

  // ── Choice selection ──────────────────────────────────────────────────────
  const handleChoiceSelect = useCallback(
    async (id: string, label: string, messageId: string) => {
      // Disable the choice cards
      setMessages((prev) =>
        prev.map((m) => (m.id === messageId ? { ...m, choiceSelected: true } : m))
      );

      const choiceMsg = messages.find((m) => m.id === messageId);
      const choiceType = choiceMsg?.choices?.type;
      const nomPoste = choiceMsg?.choices?.nom_poste;
      const convId = activeConvIdRef.current;

      if (choiceType === "element") {
        // id is always JSON with an 'action' field — dispatch accordingly
        if (!convId) return;
        try {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const parsed: Record<string, any> = JSON.parse(id);
          const action: string = parsed.action ?? "";

          if (action === "add_poste") {
            // Common poste found — lock affaire + LLM adds the poste
            lockDevisAffaire(convId, parsed.nom_affaire).catch(console.error);
            handleSend(
              `Affaire sélectionnée : "${parsed.nom_affaire}". Ajoute le poste "${parsed.nom_poste}" (num_poste: "${parsed.num_poste}") au panier.`
            );

          } else if (action === "show_elements") {
            // User chose "add separately" — create one card per element
            const elements: Array<{ text: string; occurrences: Record<string, string>[] }> = parsed.elements ?? [];
            const subOptions = elements.map((el) => {
              const occs: Record<string, string>[] = el.occurrences ?? [];
              const affLabels = occs.map((o) => o.nom_affaire).filter(Boolean);
              const detail = affLabels.length
                ? `${affLabels.length > 1 ? "Affaires" : "Affaire"} : ${affLabels.join(", ")}`
                : undefined;
              const idData =
                occs.length === 1
                  ? { ...occs[0] }
                  : { action: "show_element_affaires", element_text: el.text, occurrences: occs };
              return { id: JSON.stringify(idData), label: el.text, detail };
            });
            const subId = Math.random().toString(36).slice(2);
            const subQ = "Lequel des éléments souhaitez-vous ajouter au panier ?";
            setMessages((prev) => [
              ...prev,
              {
                id: subId, role: "assistant" as const, content: subQ,
                type: "choices" as const,
                choices: { type: "element" as const, question: subQ, options: subOptions },
                choiceSelected: false,
              },
            ]);

          } else if (action === "show_element_affaires") {
            // User chose element alone — show affaire sub-selection cards
            const occs: Record<string, string>[] = parsed.occurrences ?? [];
            const elText: string = parsed.element_text ?? label;
            const subOptions = occs.map((occ) => ({
              id: JSON.stringify({ action: "add_element", ...occ }),
              label: occ.nom_affaire || "?",
              detail: `Poste : ${occ.nom_poste}${occ.fournisseur ? ` · ${occ.fournisseur}` : ""}`,
            }));
            const subId = Math.random().toString(36).slice(2);
            const subQ = `L'élément « ${elText} » est disponible dans plusieurs affaires. Laquelle utiliser ?`;
            setMessages((prev) => [
              ...prev,
              {
                id: subId, role: "assistant" as const, content: subQ,
                type: "choices" as const,
                choices: { type: "element" as const, question: subQ, options: subOptions },
                choiceSelected: false,
              },
            ]);

          } else if (action === "add_element") {
            // Final: add element directly to panier
            const newItems = await addElementToPanierDirect(convId, parsed as Record<string, string>);
            if (newItems.length > 0) {
              setPanier((prev) => [...prev, ...newItems]);
              const confirmId = Math.random().toString(36).slice(2);
              const confirmMsg = { id: confirmId, role: "assistant" as const, content: `**${parsed.elements ?? label}** ajouté au panier.` };
              setMessages((prev) => [...prev, confirmMsg]);
              await addMessage(convId, "assistant", confirmMsg.content);
            }
          }
        } catch (err) {
          console.error("Erreur sélection élément:", err);
        }
      } else if (choiceType === "relevance") {
        // id is JSON — either a poste ({nom_poste, occurrences}) or an action ({action: ...})
        try {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const parsed: Record<string, any> = JSON.parse(id);

          if (parsed.action === "search_docs") {
            // Tell the LLM to search in docs for the original query
            handleSend(
              `Aucun résultat pertinent dans le catalogue pour « ${parsed.query} ». Cherche dans la documentation PDF avec les termes de la description technique et des composants.`
            );
          } else if (parsed.action === "refine") {
            // Just dismiss — user will retype in the input (card already disabled above)
          } else if (parsed.nom_poste) {
            // User chose to use one of the found postes.
            // Inline the same logic as type="poste" to avoid recursive dispatch.
            if (!convId) return;
            const nomPoste: string = parsed.nom_poste;
            const occurrences: Array<Record<string, string>> = parsed.occurrences ?? [];
            const uniqueAffaires = occurrences.filter(
              (occ, idx) => occurrences.findIndex((o) => o.nom_affaire === occ.nom_affaire) === idx
            );
            if (uniqueAffaires.length <= 1) {
              const occ = uniqueAffaires[0] ?? {};
              if (occ.nom_affaire) lockDevisAffaire(convId, occ.nom_affaire).catch(console.error);
              handleSend(
                `Affaire sélectionnée : "${occ.nom_affaire}". Ajoute le poste "${nomPoste}"${occ.num_poste ? ` (num_poste: "${occ.num_poste}")` : ""} au panier.`
              );
            } else {
              const subOptions = uniqueAffaires.map((occ) => {
                const detailParts = [
                  occ.ensemble && `Ensemble : ${occ.ensemble}`,
                  occ.fournisseur && `Fournisseur : ${occ.fournisseur}`,
                ].filter(Boolean);
                return {
                  id: JSON.stringify({ nom_poste: nomPoste, nom_affaire: occ.nom_affaire, num_poste: occ.num_poste }),
                  label: occ.nom_affaire,
                  detail: detailParts.join(" · ") || undefined,
                };
              });
              const subId = Math.random().toString(36).slice(2);
              const subQ = `Le poste « ${nomPoste} » existe dans plusieurs affaires. Quelle affaire utiliser ?`;
              setMessages((prev) => [
                ...prev,
                {
                  id: subId, role: "assistant" as const, content: subQ,
                  type: "choices" as const,
                  choices: { type: "poste_affaire" as const, question: subQ, options: subOptions },
                  choiceSelected: false,
                },
              ]);
            }
          }
        } catch {
          handleSend(label);
        }
        return;
      } else if (choiceType === "poste") {
        // id = JSON { nom_poste, occurrences: [{nom_affaire, num_poste, fournisseur, ensemble}] }
        // No new LLM search needed — occurrences were embedded by the backend.
        if (!convId) return;
        try {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const parsed: { nom_poste: string; occurrences: Array<Record<string, string>> } = JSON.parse(id);
          const nomPoste = parsed.nom_poste;
          const occurrences = parsed.occurrences ?? [];

          // Deduplicate by nom_affaire
          const uniqueAffaires = occurrences.filter(
            (occ, idx) => occurrences.findIndex((o) => o.nom_affaire === occ.nom_affaire) === idx
          );

          if (uniqueAffaires.length <= 1) {
            // Single affaire — lock it and send direct add instruction
            const occ = uniqueAffaires[0] ?? {};
            if (occ.nom_affaire) {
              lockDevisAffaire(convId, occ.nom_affaire).catch(console.error);
            }
            handleSend(
              `Affaire sélectionnée : "${occ.nom_affaire}". Ajoute le poste "${nomPoste}"${occ.num_poste ? ` (num_poste: "${occ.num_poste}")` : ""} au panier.`
            );
          } else {
            // Multiple affaires — show sub-choice cards without a new LLM call
            const subOptions = uniqueAffaires.map((occ) => {
              const detailParts = [
                occ.ensemble && `Ensemble : ${occ.ensemble}`,
                occ.fournisseur && `Fournisseur : ${occ.fournisseur}`,
              ].filter(Boolean);
              return {
                id: JSON.stringify({ nom_poste: nomPoste, nom_affaire: occ.nom_affaire, num_poste: occ.num_poste }),
                label: occ.nom_affaire,
                detail: detailParts.join(" · ") || undefined,
              };
            });
            const subId = Math.random().toString(36).slice(2);
            const subQ = `Le poste « ${nomPoste} » existe dans plusieurs affaires. Quelle affaire utiliser pour ce devis ?`;
            setMessages((prev) => [
              ...prev,
              {
                id: subId,
                role: "assistant" as const,
                content: subQ,
                type: "choices" as const,
                choices: { type: "poste_affaire" as const, question: subQ, options: subOptions },
                choiceSelected: false,
              },
            ]);
          }
        } catch {
          // Fallback if id is not valid JSON
          handleSend(`Poste sélectionné : "${label}". Recherche ce poste exact et ajoute-le au panier.`);
        }
      } else if (choiceType === "poste_affaire") {
        // id = JSON { nom_poste, nom_affaire, num_poste }
        // Affaire sub-selection after poste pick — lock affaire then add directly.
        if (!convId) return;
        try {
          const parsed: { nom_poste: string; nom_affaire: string; num_poste: string } = JSON.parse(id);
          lockDevisAffaire(convId, parsed.nom_affaire).catch(console.error);
          handleSend(
            `Affaire sélectionnée : "${parsed.nom_affaire}". Ajoute le poste "${parsed.nom_poste}"${parsed.num_poste ? ` (num_poste: "${parsed.num_poste}")` : ""} au panier.`
          );
        } catch {
          handleSend(label);
        }
      } else if (choiceType === "search_column") {
        // No nom_poste matched — user picked a column to search in.
        try {
          const parsed: { action: string; column?: string; query?: string } = JSON.parse(id);
          if (parsed.action === "search_column" && parsed.column && parsed.query) {
            const colLabels: Record<string, string> = {
              elements: "sous-composants (éléments)",
              ensemble: "type d'ensemble",
              nom_affaire: "nom d'affaire",
              fournisseur: "fournisseur",
            };
            const colLabel = colLabels[parsed.column] ?? parsed.column;
            handleSend(
              `Cherche "${parsed.query}" dans la colonne ${colLabel}. Appelle search_catalog avec column="${parsed.column}".`
            );
          }
          // "refine" → user will retype, nothing to do
        } catch {
          // ignore
        }
      } else if (choiceType === "search_scope") {
        // After add_to_panier: user picks whether next search stays in same affaire or full catalog.
        if (!convId) return;
        try {
          const parsed: { action: string } = JSON.parse(id);
          if (parsed.action === "search_all_affaires") {
            await setSearchScope(convId, true);
          }
          // "search_same_affaire" → nothing to do, search_all stays false (already reset by backend)
          scopeChoicePendingRef.current = false;
          const savedMsg = pendingMessageRef.current;
          pendingMessageRef.current = "";
          if (savedMsg) handleSend(savedMsg);
        } catch {
          // ignore parse errors
        }
      } else {
        // type === "affaire" (or legacy undefined): id/label = nom_affaire.
        // Lock the affaire in the backend immediately before the next LLM call.
        if (convId) {
          lockDevisAffaire(convId, id).catch(console.error);
        }
        // Send a precise instruction: LLM calls add_to_panier directly, no new search needed.
        const userMsg = nomPoste
          ? `Affaire sélectionnée : "${label}". Ajoute le poste "${nomPoste}" au panier.`
          : label;
        handleSend(userMsg);
      }
    },
    [handleSend, messages]
  );

  // ── Generate devis ────────────────────────────────────────────────────────
  const handleGenerateDevis = useCallback(async () => {
    const convId = activeConvIdRef.current;
    if (!convId || isGenerating) return;

    setIsGenerating(true);
    setError(null);

    const assistantId = Math.random().toString(36).slice(2);
    let devisContent = "";

    try {
      let firstToken = true;

      for await (const event of streamGenerateDevis(convId)) {
        if (event.error) {
          setError(event.error);
          break;
        }

        if (event.token) {
          devisContent += event.token;
          if (firstToken) {
            firstToken = false;
            setMessages((prev) => [
              ...prev,
              { id: assistantId, role: "assistant", content: devisContent, isStreaming: true },
            ]);
          } else {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: devisContent } : m
              )
            );
          }
          scrollToBottom();
        }

        if (event.done) {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, isStreaming: false } : m))
          );
          break;
        }
      }

      if (devisContent && convId) {
        await addMessage(convId, "assistant", devisContent);
        refreshConversations();
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Erreur lors de la génération";
      setError(msg);
    } finally {
      setIsGenerating(false);
    }
  }, [isGenerating, scrollToBottom, refreshConversations]);

  // ── Export Excel ──────────────────────────────────────────────────────────
  const handleExportExcel = useCallback(async () => {
    const convId = activeConvIdRef.current;
    if (!convId || isExporting) return;
    setIsExporting(true);
    setError(null);
    try {
      await exportDevisExcel(convId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur lors de l'export Excel");
    } finally {
      setIsExporting(false);
    }
  }, [isExporting]);

  // ── Panier actions ────────────────────────────────────────────────────────
  const handleRemoveItem = useCallback(
    async (itemId: string) => {
      const convId = activeConvIdRef.current;
      if (!convId) return;
      await removePanierItem(convId, itemId);
      setPanier((prev) => prev.filter((i) => i.id !== itemId));
    },
    []
  );

  const handleClearPanier = useCallback(async () => {
    const convId = activeConvIdRef.current;
    if (!convId) return;
    await clearPanier(convId);
    setPanier([]);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isWelcome = messages.length === 0 && !isLoading;

  const inputBar = (
    <>
      <div className="flex items-end gap-3 bg-card border border-border rounded-2xl px-4 py-3 shadow-sm">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Décrivez les specs ou entrez des références…"
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
        {!catalogLoaded && (
          <p className="text-fluid-xs text-amber-500/80">
            Catalogue non chargé — lookup désactivé
          </p>
        )}
      </div>
    </>
  );

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Main chat column ─────────────────────────────────────────────── */}
      <div className="relative flex flex-col flex-1 min-w-0 bg-background">
        {/* Header */}
        <header className="flex items-center gap-3 h-14 px-4 border-b border-border flex-shrink-0">
          <SidebarTrigger className="text-muted-foreground hover:text-foreground" />
          <span className="text-sm font-medium text-muted-foreground">Devis de synthèse</span>
        </header>

        {/* LLM loading banner */}
        {!llmReady && !error && (
          <div className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 text-sm text-amber-600 dark:text-amber-400">
            <svg className="animate-spin h-3.5 w-3.5 shrink-0" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
            Modèle IA en cours de chargement…
          </div>
        )}

        {isWelcome ? (
          /* ── WELCOME ───────────────────────────────────────────────────── */
          <div className="flex-1 overflow-y-auto flex flex-col items-center px-4 pt-[10vh] md:pt-[14vh]">
            <h1 className="text-fluid-2xl font-semibold text-foreground mb-2">
              Devis de synthèse
            </h1>
            <p className="text-fluid-sm text-muted-foreground mb-8 text-center max-w-md">
              Décrivez les spécifications techniques ou entrez des références directes. Je recherche dans le catalogue et les documents pour composer votre devis.
            </p>

            <div className="grid grid-cols-2 gap-3 w-full max-w-[680px] mb-5">
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

            <div className="w-full max-w-[680px]">{inputBar}</div>
          </div>
        ) : (
          /* ── CHAT ──────────────────────────────────────────────────────── */
          <>
            <div
              ref={scrollAreaRef}
              onScroll={handleScroll}
              className="flex-1 overflow-y-auto"
            >
              <div className="max-w-[680px] w-full mx-auto px-4 md:px-6 py-6 md:py-10">
                {messages.map((msg) => (
                  <MessageItem
                    key={msg.id}
                    msg={msg}
                    toolCalls={msg.role === "assistant" ? messageToolCalls[msg.id] : undefined}
                    onChoiceSelect={handleChoiceSelect}
                  />
                ))}
                {/* Active tool calls for the in-progress turn */}
                {isLoading && activeToolCalls.length > 0 && (
                  <div className="flex gap-3 mb-4">
                    <div className="w-8 shrink-0" />
                    <div>
                      {activeToolCalls.map((tc) => (
                        <ToolCallBadge key={tc.id} tool={tc} />
                      ))}
                    </div>
                  </div>
                )}
                {isLoading && activeToolCalls.length === 0 && <LoadingDots />}
                {error && (
                  <p className="text-center text-sm text-destructive py-2">{error}</p>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>

            {showScrollBtn && (
              <button
                onClick={() => scrollToBottom()}
                className="absolute bottom-28 right-6 w-9 h-9 rounded-full bg-card border border-border shadow-md flex items-center justify-center text-muted-foreground hover:text-foreground transition-all"
              >
                <ArrowDown size={16} />
              </button>
            )}

            <div className="flex-shrink-0 px-4 pb-4 pt-2 bg-background">
              <div className="max-w-[680px] mx-auto">{inputBar}</div>
            </div>
          </>
        )}
      </div>

      {/* ── Panier panel (visible only when items exist) ─────────────────── */}
      {panier.length > 0 && (
        <PanierPanel
          panier={panier}
          onRemoveItem={handleRemoveItem}
          onClear={handleClearPanier}
          onGenerate={handleGenerateDevis}
          onExportExcel={handleExportExcel}
          isGenerating={isGenerating}
          isExporting={isExporting}
        />
      )}
    </div>
  );
}
