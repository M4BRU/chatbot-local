"use client";

import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { ArrowDown, ChevronDown, ChevronRight, Download, MoreHorizontal, Send, Trash2, X } from "lucide-react";
import Image from "next/image";
import { AssistantMessage, LoadingDots, UserBubble } from "@/components/chat/MarkdownMessage";
import {
  addPosteToPanierDirect,
  addMessage,
  clearPanier,
  exportDevisExcel,
  fetchCatalogStatus,
  fetchCollections,
  fetchDevisSettings,
  fetchLLMStatus,
  fetchPanier,
  fetchPosteElements,
  getConversationMessages,
  lockDevisAffaire,
  removePanierItem,
  streamDevisChat,
  streamGenerateDevis,
  updateDevisSettings,
  updatePanierItem,
} from "@/app/lib/api";
import type { CatalogElement, ChatMessage, PanierItem, RfqCandidate } from "@/app/lib/types";
import { useConversation } from "@/app/providers";
import { SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { useChoiceHandler } from "./useChoiceHandler";

// ─── Shared spinner ──────────────────────────────────────────────────────────
function Spinner({ className }: { className?: string }) {
  return (
    <svg className={cn("animate-spin shrink-0", className)} fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  );
}

// ─── RFQ Planning progress banner ─────────────────────────────────────────────
const RFQ_STATUS_LABELS: Record<string, string> = {
  analyzing:    "Analyse du RFQ",
  searching:    "Recherche documentaire",
  gap_check:    "Vérification des lacunes",
  synthesizing: "Synthèse du contexte",
};

function RfqPlanningBanner({
  planning,
}: {
  planning: { status: string; step: string; dimensions_found?: number };
}) {
  return (
    <div className="flex gap-3 mb-4">
      <div className="w-8 shrink-0" />
      <div className="flex-1 rounded-lg border border-blue-500/20 bg-blue-500/5 px-4 py-3 max-w-[580px]">
        <div className="flex items-center gap-2 mb-1.5">
          <Spinner className="h-3.5 w-3.5 text-blue-500" />
          <span className="text-xs font-semibold text-blue-600 dark:text-blue-400">
            Analyse RFQ — {RFQ_STATUS_LABELS[planning.status] ?? planning.status}
          </span>
          {planning.dimensions_found != null && planning.dimensions_found > 0 && (
            <span className="ml-auto text-xs bg-blue-500/15 rounded-full px-2 py-0.5 text-blue-700 dark:text-blue-300 shrink-0">
              {planning.dimensions_found} dim.
            </span>
          )}
        </div>
        <p className="text-xs text-muted-foreground">{planning.step}</p>
      </div>
    </div>
  );
}

// ─── Search Workspace Modal ────────────────────────────────────────────────────
type SWPhase = "config" | "searching" | "awaiting_choice" | "done";
type SWComp = { id: string; name: string; selected: boolean; result: "pending" | "found" | "skipped" };
type SWChoice = { id: string; label: string; detail?: string };
type SearchWorkspaceState = {
  question: string;
  phase: SWPhase;
  allComps: SWComp[];
  queue: string[];       // ids of selected comps, set at launch
  currentIdx: number;    // index into queue
  choices: SWChoice[];   // results for current component
};

function SearchWorkspaceModal({
  ws,
  onUpdate,
  onClose,
}: {
  ws: SearchWorkspaceState;
  onUpdate: (ws: SearchWorkspaceState) => void;
  onClose: () => void;
}) {
  const { phase, allComps, queue, currentIdx, choices } = ws;
  const currentId = queue[currentIdx];
  const currentComp = allComps.find(c => c.id === currentId);
  const canClose = phase === "config" || phase === "done";
  const foundCount = allComps.filter(c => c.result === "found").length;
  const skippedCount = allComps.filter(c => c.result === "skipped").length;

  // Mock: simulate search delay → show results
  useEffect(() => {
    if (phase !== "searching") return;
    const t = setTimeout(() => {
      onUpdate({
        ...ws,
        phase: "awaiting_choice",
        choices: [
          { id: "m1", label: "VLM-2024-001", detail: "Affaire Paris — 3 occurrences" },
          { id: "m2", label: "VLM-2023-047", detail: "Affaire Lyon — 1 occurrence" },
          { id: "m3", label: "VLM-2024-089", detail: "Affaire Bordeaux — 2 occurrences" },
        ],
      });
    }, 1200);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, currentIdx]);

  const advance = (updatedComps: SWComp[]) => {
    const nextIdx = currentIdx + 1;
    if (nextIdx >= queue.length) {
      onUpdate({ ...ws, allComps: updatedComps, phase: "done", choices: [] });
    } else {
      onUpdate({ ...ws, allComps: updatedComps, phase: "searching", currentIdx: nextIdx, choices: [] });
    }
  };

  const handlePickChoice = (choice: SWChoice) => {
    void choice;
    const updated = allComps.map(c => c.id === currentId ? { ...c, result: "found" as const } : c);
    advance(updated);
  };

  const handleSkip = () => {
    const updated = allComps.map(c => c.id === currentId ? { ...c, result: "skipped" as const } : c);
    advance(updated);
  };

  const handleLaunch = () => {
    const selectedIds = allComps.filter(c => c.selected).map(c => c.id);
    if (selectedIds.length === 0) return;
    onUpdate({ ...ws, queue: selectedIds, currentIdx: 0, phase: "searching", choices: [] });
  };

  const resultIcon = (comp: SWComp) => {
    if (comp.result === "found") return <span className="text-green-500 text-xs leading-none">✓</span>;
    if (comp.result === "skipped") return <span className="text-muted-foreground text-xs leading-none">—</span>;
    if (comp.id === currentId && (phase === "searching" || phase === "awaiting_choice")) {
      return (
        <Spinner className="h-3 w-3 text-primary" />
      );
    }
    return <span className="w-3 h-3 rounded-full border border-border/60 inline-block shrink-0" />;
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={canClose ? onClose : undefined}
    >
      <div
        className="bg-card border border-border rounded-2xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden flex flex-col max-h-[85vh]"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border flex-shrink-0">
          <div className="flex items-center gap-2">
            {(phase === "searching" || phase === "awaiting_choice") && (
              <Spinner className="h-3.5 w-3.5 text-primary" />
            )}
            <h3 className="font-medium text-sm">
              {phase === "config" && "Composants identifiés"}
              {(phase === "searching" || phase === "awaiting_choice") && `Composant ${currentIdx + 1} / ${queue.length}`}
              {phase === "done" && "Recherche terminée"}
            </h3>
          </div>
          {canClose && (
            <button onClick={onClose} className="text-muted-foreground hover:text-foreground transition-colors">
              <X size={15} />
            </button>
          )}
        </div>

        {/* Question context */}
        <div className="px-5 py-2.5 border-b border-border/50 bg-muted/20 flex-shrink-0">
          <p className="text-xs text-muted-foreground italic line-clamp-2">{ws.question}</p>
        </div>

        {/* Component list */}
        <div className="flex-1 overflow-y-auto min-h-0">
          {phase === "config" ? (
            <div className="p-3 space-y-0.5">
              {allComps.map(comp => (
                <div key={comp.id} className="flex items-center gap-2.5 px-2 py-1.5 rounded-lg hover:bg-muted/50 transition-colors group">
                  <input
                    type="checkbox"
                    checked={comp.selected}
                    onChange={() => onUpdate({
                      ...ws,
                      allComps: allComps.map(c => c.id === comp.id ? { ...c, selected: !c.selected } : c),
                    })}
                    className="h-4 w-4 rounded accent-primary cursor-pointer"
                  />
                  <span className={cn("flex-1 text-sm", !comp.selected && "text-muted-foreground line-through")}>
                    {comp.name}
                  </span>
                  <button
                    onClick={() => onUpdate({ ...ws, allComps: allComps.filter(c => c.id !== comp.id) })}
                    className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive transition-all"
                  >
                    <X size={13} />
                  </button>
                </div>
              ))}
              {allComps.length === 0 && (
                <p className="text-xs text-muted-foreground text-center py-4">Aucun composant</p>
              )}
            </div>
          ) : (
            <div className="p-3 space-y-0.5">
              {allComps.filter(c => queue.includes(c.id)).map(comp => (
                <div
                  key={comp.id}
                  className={cn(
                    "flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors",
                    comp.id === currentId && phase !== "done"
                      ? "bg-primary/5 border border-primary/20 font-medium text-foreground"
                      : comp.result !== "pending"
                      ? "text-muted-foreground"
                      : "text-muted-foreground/50"
                  )}
                >
                  <span className="flex items-center justify-center w-4 shrink-0">{resultIcon(comp)}</span>
                  <span className="flex-1 truncate">{comp.name}</span>
                  {comp.result === "found" && <span className="text-xs text-green-600 dark:text-green-400 shrink-0">Ajouté</span>}
                  {comp.result === "skipped" && <span className="text-xs text-muted-foreground shrink-0">Passé</span>}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Results zone — awaiting_choice */}
        {phase === "awaiting_choice" && choices.length > 0 && (
          <div className="border-t border-border flex-shrink-0">
            <p className="text-xs font-medium text-muted-foreground px-4 pt-3 pb-2">
              Résultats pour <span className="text-foreground">&ldquo;{currentComp?.name}&rdquo;</span>
            </p>
            <div className="px-3 pb-3 space-y-1.5 max-h-48 overflow-y-auto">
              {choices.map(c => (
                <button
                  key={c.id}
                  onClick={() => handlePickChoice(c)}
                  className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg border border-border hover:border-primary/40 hover:bg-primary/5 transition-all text-left group"
                >
                  <div>
                    <p className="text-sm font-medium text-foreground">{c.label}</p>
                    {c.detail && <p className="text-xs text-muted-foreground">{c.detail}</p>}
                  </div>
                  <ChevronRight size={14} className="text-muted-foreground group-hover:text-primary transition-colors shrink-0" />
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="px-4 pb-4 pt-2.5 border-t border-border flex-shrink-0">
          {phase === "config" && (
            <button
              onClick={handleLaunch}
              disabled={allComps.filter(c => c.selected).length === 0}
              className="w-full bg-primary text-primary-foreground rounded-lg py-2 text-sm font-medium disabled:opacity-50 hover:bg-primary/90 transition-colors"
            >
              Lancer la recherche ({allComps.filter(c => c.selected).length} composant{allComps.filter(c => c.selected).length > 1 ? "s" : ""})
            </button>
          )}
          {phase === "searching" && (
            <p className="text-xs text-muted-foreground text-center py-1">Recherche en cours…</p>
          )}
          {phase === "awaiting_choice" && (
            <button
              onClick={handleSkip}
              className="w-full border border-border rounded-lg py-2 text-sm text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            >
              Passer ce composant
            </button>
          )}
          {phase === "done" && (
            <div className="flex items-center gap-3">
              <p className="flex-1 text-xs text-muted-foreground">
                <span className="text-green-600 dark:text-green-400 font-medium">{foundCount} trouvé{foundCount > 1 ? "s" : ""}</span>
                {skippedCount > 0 && <span> · {skippedCount} passé{skippedCount > 1 ? "s" : ""}</span>}
              </p>
              <button
                onClick={onClose}
                className="bg-primary text-primary-foreground rounded-lg px-4 py-2 text-sm font-medium hover:bg-primary/90 transition-colors"
              >
                Fermer
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Tool call status badge ────────────────────────────────────────────────────
const TOOL_LABELS: Record<string, string> = {
  search_catalog: "Recherche dans le catalogue",
  search_docs: "Recherche dans les documents",
  report_findings: "Identification des composants",
  add_to_panier: "Ajout au devis",
  ask_user_choice: "Présentation des options",
  set_devis_settings: "Mise à jour des coefficients",
  update_panier_item: "Modification du poste",
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
        <Spinner className="h-3 w-3" />
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

// ─── Affaire row with lazy-loaded elements ─────────────────────────────────────
function AffaireRow({
  nomPoste,
  nomAffaire,
  prixTotal,
  disabled,
}: {
  nomPoste: string;
  nomAffaire: string;
  prixTotal: string | null;
  disabled?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const [elements, setElements] = useState<CatalogElement[] | null>(null);
  const [loading, setLoading] = useState(false);

  const toggle = async () => {
    if (disabled) return;
    if (!expanded && elements === null) {
      setLoading(true);
      try {
        const data = await fetchPosteElements(nomPoste, nomAffaire);
        setElements(data);
      } catch {
        setElements([]);
      } finally {
        setLoading(false);
      }
    }
    setExpanded((v) => !v);
  };

  return (
    <div className="border-b border-border/40 last:border-0">
      <button
        onClick={toggle}
        disabled={disabled}
        className="w-full flex items-center justify-between px-3 py-2 text-xs hover:bg-muted/30 transition-colors text-left disabled:opacity-50"
      >
        <span className="font-medium text-foreground truncate max-w-[60%]">{nomAffaire}</span>
        <div className="flex items-center gap-2 shrink-0">
          {prixTotal && <span className="text-muted-foreground">{prixTotal}</span>}
          {loading ? (
            <Spinner className="h-3 w-3 text-muted-foreground" />
          ) : (
            <ChevronDown className={cn("h-3 w-3 text-muted-foreground transition-transform", expanded && "rotate-180")} />
          )}
        </div>
      </button>

      {expanded && elements !== null && (
        <div className="bg-muted/20 border-t border-border/30 overflow-x-auto">
          {elements.length === 0 ? (
            <p className="px-3 py-2 text-xs text-muted-foreground">Aucun élément trouvé.</p>
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border/40 bg-muted/40">
                  <th className="px-3 py-1.5 text-left text-muted-foreground font-medium">Éléments</th>
                  <th className="px-3 py-1.5 text-left text-muted-foreground font-medium">Fournisseur</th>
                  <th className="px-3 py-1.5 text-left text-muted-foreground font-medium">Prix €</th>
                </tr>
              </thead>
              <tbody>
                {elements.map((el, i) => (
                  <tr key={i} className={cn("border-b border-border/30 last:border-0", i % 2 !== 0 && "bg-muted/10")}>
                    <td className="px-3 py-1.5 text-foreground">{el.elements || "—"}</td>
                    <td className="px-3 py-1.5 text-foreground">{el.fournisseur || "—"}</td>
                    <td className="px-3 py-1.5 text-foreground">{el.fourniture != null ? String(el.fourniture) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}

function QueryVariantSelector({
  question,
  options,
  onSelect,
  disabled,
  type,
}: {
  question: string;
  options: VariantOption[];
  onSelect: (id: string, label: string) => void;
  disabled?: boolean;
  type?: string;
}) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  return (
    <div className="max-w-2xl w-full">
      <p className="text-sm text-foreground mb-3">{question}</p>

      <div className="flex flex-col gap-2">
        {options.map((opt, i) => {
          const isExpanded = expandedIdx === i;

          // ── "poste" type: each card lists affaires, each affaire row is expandable ──
          if (type === "poste") {
            let nomPoste = opt.label;
            let occurrences: { nom_affaire: string; num_poste: string }[] = [];
            try {
              const parsed = JSON.parse(opt.id);
              nomPoste = parsed.nom_poste ?? opt.label;
              occurrences = parsed.occurrences ?? [];
            } catch { /* keep defaults */ }

            // Build prix map from opt.rows (col 0 = nom_affaire, col 1 = prix)
            const prixMap: Record<string, string> = {};
            (opt.rows ?? []).forEach((row) => {
              if (row[0]) prixMap[String(row[0])] = row[1] != null ? String(row[1]) : "";
            });
            const uniqueAffaires = [...new Set(occurrences.map((o) => o.nom_affaire))];

            return (
              <div
                key={opt.id}
                className={cn("rounded-lg border bg-card transition-all", isExpanded ? "border-foreground/40 shadow-sm" : "border-border")}
              >
                {/* Card header */}
                <div className="p-3">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-medium text-foreground">{nomPoste}</p>
                    {opt.detail && <span className="text-xs text-muted-foreground shrink-0">{opt.detail}</span>}
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <button
                      onClick={() => setExpandedIdx(isExpanded ? null : i)}
                      className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <ChevronDown className={cn("h-3 w-3 transition-transform", isExpanded && "rotate-180")} />
                      {isExpanded ? "Masquer les affaires" : `${uniqueAffaires.length} affaire${uniqueAffaires.length > 1 ? "s" : ""}`}
                    </button>
                    {!disabled && (
                      <button
                        onClick={() => onSelect(opt.id, opt.label)}
                        className="ml-auto rounded-full bg-foreground text-background px-3 py-1 text-xs font-medium hover:opacity-80 transition-all"
                      >
                        Sélectionner
                      </button>
                    )}
                  </div>
                </div>

                {/* Affaire rows — each expandable to show elements */}
                {isExpanded && (
                  <div className="border-t border-border">
                    {uniqueAffaires.map((aff) => (
                      <AffaireRow
                        key={aff}
                        nomPoste={nomPoste}
                        nomAffaire={aff}
                        prixTotal={prixMap[aff] ?? null}
                        disabled={disabled}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          }

          // ── "affaire" type: each card has pre-loaded elements ──────────────────
          const hasTable = opt.columns && opt.rows && opt.rows.length > 0;
          return (
            <div
              key={opt.id}
              className={cn("rounded-lg border bg-card transition-all", isExpanded ? "border-foreground/40 shadow-sm" : "border-border")}
            >
              <div className="p-3">
                <p className="text-xs font-medium text-foreground">{opt.label}</p>
                <div className="flex items-center gap-2 mt-1 flex-wrap">
                  {opt.description && (
                    <span className="text-xs bg-foreground/10 text-foreground/70 rounded-full px-2 py-0.5">{opt.description}</span>
                  )}
                  {opt.detail && <span className="text-xs text-muted-foreground">{opt.detail}</span>}
                </div>
                <div className="flex items-center gap-2 mt-2 flex-wrap">
                  {hasTable && (
                    <button
                      onClick={() => setExpandedIdx(isExpanded ? null : i)}
                      className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <ChevronDown className={cn("h-3 w-3 transition-transform", isExpanded && "rotate-180")} />
                      {isExpanded ? "Masquer les éléments" : `Voir les ${opt.rows!.length} éléments`}
                    </button>
                  )}
                  {!disabled && (
                    <button
                      onClick={() => onSelect(opt.id, opt.label)}
                      className="ml-auto rounded-full bg-foreground text-background px-3 py-1 text-xs font-medium hover:opacity-80 transition-all"
                    >
                      Sélectionner
                    </button>
                  )}
                </div>
              </div>

              {isExpanded && hasTable && (
                <div className="border-t border-border overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-border bg-muted/50">
                        {opt.columns!.map((col) => (
                          <th key={col} className="px-3 py-2 text-left text-muted-foreground font-medium whitespace-nowrap">{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {opt.rows!.map((row, ri) => (
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
            </div>
          );
        })}
      </div>
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
  if (msg.role === "user") return <UserBubble content={msg.content} />;

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
            type={choiceType}
          />
        </div>
      );
    }

    // Findings confirmation card — list of components found in docs
    if (choiceType === "findings_confirmation") {
      const components = msg.choices.components ?? [];
      const catalogMatches = msg.choices.catalog_matches ?? [];
      const docOnlyModels = msg.choices.doc_only_models ?? [];
      const hasSpecInfo = catalogMatches.length > 0;

      const specIcon: Record<string, string> = {
        match: "✅",
        partial: "🟡",
        no_match: "❌",
        unknown: "❓",
      };

      return (
        <div className="flex gap-3 mb-6">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center mt-0.5 border border-border">
            <Image src="/logoVLM.png" alt="Devis" width={18} height={18} className="object-contain" />
          </div>
          <div className="max-w-xl w-full">
            <p className="text-sm text-foreground mb-3">{msg.choices.question}</p>

            {/* Spec-aware: catalog matches with status */}
            {hasSpecInfo && (
              <div className="rounded-lg border border-border bg-muted/20 mb-3 overflow-hidden">
                <p className="text-xs font-semibold text-muted-foreground px-3 py-2 border-b border-border/60 uppercase tracking-wide">
                  Postes catalogue
                </p>
                {catalogMatches.map((m, i) => (
                  <div key={i} className="flex items-start gap-2 px-3 py-2 border-b border-border/40 last:border-0">
                    <span className="text-sm shrink-0">{specIcon[m.spec_status] ?? "❓"}</span>
                    <div>
                      <span className="text-xs font-medium text-foreground">{m.nom_poste}</span>
                      {m.note && <p className="text-xs text-muted-foreground mt-0.5">{m.note}</p>}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Standard flow: simple component chips */}
            {!hasSpecInfo && components.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mb-4">
                {components.map((comp, i) => (
                  <span
                    key={i}
                    className="rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-700 dark:text-blue-300 px-2.5 py-1 text-xs font-medium"
                  >
                    {comp}
                  </span>
                ))}
              </div>
            )}

            {/* Doc-only models — informational */}
            {docOnlyModels.length > 0 && (
              <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 mb-3 px-3 py-2">
                <p className="text-xs font-semibold text-amber-600 dark:text-amber-400 mb-1">
                  📄 Modèles en documentation (absents du catalogue)
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {docOnlyModels.map((m, i) => (
                    <span key={i} className="text-xs text-amber-700 dark:text-amber-300 bg-amber-500/10 rounded-full px-2 py-0.5 border border-amber-500/20">
                      {m}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Action buttons */}
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
    <AssistantMessage content={msg.content} isStreaming={msg.isStreaming} noCode>
      {toolCalls && toolCalls.map((tc) => <ToolCallBadge key={tc.id} tool={tc} />)}
    </AssistantMessage>
  );
}

// ─── Poste row (editable MdO fields, expandable elements) ─────────────────────
function PosteRow({
  item,
  isOption,
  highlighted,
  onUpdate,
  onRemove,
}: {
  item: PanierItem;
  isOption: boolean;
  highlighted?: boolean;
  onUpdate: (itemId: string, field: string, value: number | boolean) => Promise<void>;
  onRemove: (itemId: string) => void;
}) {
  const etudRef = useRef<HTMLInputElement>(null);
  const atelRef = useRef<HTMLInputElement>(null);
  const clientRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  const [elements, setElements] = useState<CatalogElement[] | null>(null);
  const [loadingEl, setLoadingEl] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [pending, setPending] = useState<Record<string, boolean>>({});
  const [fieldErr, setFieldErr] = useState<Record<string, boolean>>({});

  // Sync uncontrolled inputs when item props change externally (e.g., LLM update_panier_item)
  useEffect(() => {
    if (etudRef.current && document.activeElement !== etudRef.current)
      etudRef.current.value = String(item.nbre_jours_etude ?? 0);
    if (atelRef.current && document.activeElement !== atelRef.current)
      atelRef.current.value = String(item.nbre_jours_atelier ?? 0);
    if (clientRef.current && document.activeElement !== clientRef.current)
      clientRef.current.value = String(item.nbre_jours_client ?? 0);
  }, [item.nbre_jours_etude, item.nbre_jours_atelier, item.nbre_jours_client]);

  // Close menu on outside click
  useEffect(() => {
    if (!menuOpen) return;
    const handle = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, [menuOpen]);

  const toggleExpand = async () => {
    if (!expanded && elements === null) {
      setLoadingEl(true);
      try {
        const data = await fetchPosteElements(item.nom_poste, item.nom_affaire);
        setElements(data);
      } catch {
        setElements([]);
      } finally {
        setLoadingEl(false);
      }
    }
    setExpanded(v => !v);
  };

  const handleNumBlur = async (field: string, ref: React.RefObject<HTMLInputElement | null>) => {
    const el = ref.current;
    if (!el) return;
    const newVal = Math.max(0, parseInt(el.value) || 0);
    const originalVal = (item[field as keyof PanierItem] as number) ?? 0;
    if (newVal === originalVal) return;
    setPending(prev => ({ ...prev, [field]: true }));
    try {
      await onUpdate(item.id, field, newVal);
    } catch {
      el.value = String(originalVal);
      setFieldErr(prev => ({ ...prev, [field]: true }));
      setTimeout(() => setFieldErr(prev => ({ ...prev, [field]: false })), 3000);
    } finally {
      setPending(prev => ({ ...prev, [field]: false }));
    }
  };

  const numFields: [string, React.RefObject<HTMLInputElement | null>][] = [
    ["nbre_jours_etude", etudRef],
    ["nbre_jours_atelier", atelRef],
    ["nbre_jours_client", clientRef],
  ];

  return (
    <>
      <tr
        onClick={toggleExpand}
        className={cn(
          "border-b border-border/40 cursor-pointer select-none transition-colors",
          isOption ? "bg-amber-500/5 hover:bg-amber-500/10" : "hover:bg-muted/20",
          highlighted && "bg-blue-500/10 animate-pulse"
        )}
      >
        {/* Expand indicator (visual only — row click handles toggle) */}
        <td className="px-1.5 py-1.5 text-center w-6 text-muted-foreground">
          {loadingEl ? (
            <Spinner className="h-3 w-3 inline" />
          ) : expanded ? (
            <ChevronDown className="h-3 w-3 inline" />
          ) : (
            <ChevronRight className="h-3 w-3 inline" />
          )}
        </td>
        {/* Poste + Affaire */}
        <td className="px-2 py-1.5 text-xs max-w-[120px]">
          <div className="font-medium truncate" title={item.nom_poste}>{item.nom_poste}</div>
          {item.nom_affaire && (
            <div className="text-muted-foreground truncate text-[10px]" title={item.nom_affaire}>
              {item.nom_affaire}
            </div>
          )}
        </td>
        {/* MdO number fields — stopPropagation so clicking input doesn't toggle row */}
        {numFields.map(([field, ref]) => (
          <td key={field} className="px-1 py-1.5 text-center" onClick={e => e.stopPropagation()}>
            <input
              ref={ref}
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              defaultValue={String((item[field as keyof PanierItem] as number) ?? 0)}
              onBlur={() => handleNumBlur(field, ref)}
              disabled={!!pending[field]}
              className={cn(
                "w-10 text-center text-xs border rounded px-1 py-0.5 bg-transparent focus:outline-none focus:ring-1 focus:ring-foreground/20",
                pending[field] && "opacity-50 cursor-wait",
                fieldErr[field] ? "border-destructive" : "border-border"
              )}
            />
          </td>
        ))}
        {/* ··· context menu */}
        <td className="px-1 py-1.5 text-center" onClick={e => e.stopPropagation()}>
          <div ref={menuRef} className="relative inline-block">
            <button
              onClick={() => setMenuOpen(v => !v)}
              className="text-muted-foreground hover:text-foreground transition-colors"
              title="Actions"
            >
              <MoreHorizontal className="h-3.5 w-3.5" />
            </button>
            {menuOpen && (
              <div className="absolute right-0 bottom-full mb-1 z-50 bg-card border border-border rounded-md shadow-lg py-1 min-w-[180px]">
                <button
                  onClick={() => {
                    onUpdate(item.id, "is_option", !isOption).catch(() => {});
                    setMenuOpen(false);
                  }}
                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-muted/60 transition-colors"
                >
                  {isOption ? "↑ Remettre en poste principal" : "↓ Passer en option"}
                </button>
                <div className="border-t border-border/50 my-1" />
                <button
                  onClick={() => { onRemove(item.id); setMenuOpen(false); }}
                  className="w-full text-left px-3 py-1.5 text-xs text-destructive hover:bg-destructive/10 transition-colors"
                >
                  Supprimer
                </button>
              </div>
            )}
          </div>
        </td>
        {/* Delete shortcut */}
        <td className="px-1.5 py-1.5 text-center" onClick={e => e.stopPropagation()}>
          <button
            onClick={() => onRemove(item.id)}
            className="text-muted-foreground hover:text-destructive transition-colors"
            title="Retirer"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </td>
      </tr>
      {/* Expanded elements sub-table */}
      {expanded && elements !== null && (
        <tr className="border-b border-border/40">
          <td />
          <td colSpan={6} className="pb-2 pr-2">
            <div className="rounded border border-border/40 bg-muted/20 overflow-x-auto">
              {elements.length === 0 ? (
                <p className="px-3 py-2 text-xs text-muted-foreground">Aucun élément trouvé.</p>
              ) : (
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border/40 bg-muted/30">
                      <th className="px-2 py-1 text-left text-muted-foreground font-medium">Ensemble</th>
                      <th className="px-2 py-1 text-left text-muted-foreground font-medium">Éléments</th>
                      <th className="px-2 py-1 text-left text-muted-foreground font-medium">Fournisseur</th>
                      <th className="px-2 py-1 text-right text-muted-foreground font-medium">Fourniture €</th>
                    </tr>
                  </thead>
                  <tbody>
                    {elements.map((el, i) => (
                      <tr key={i} className={cn("border-b border-border/30 last:border-0", i % 2 !== 0 && "bg-muted/10")}>
                        <td className="px-2 py-1">{el.ensemble || "—"}</td>
                        <td className="px-2 py-1">{el.elements || "—"}</td>
                        <td className="px-2 py-1">{el.fournisseur || "—"}</td>
                        <td className="px-2 py-1 text-right">{el.fourniture != null ? String(el.fourniture) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// ─── Devis panel ──────────────────────────────────────────────────────────────
function DevisPanel({
  postes,
  devisSettings,
  highlightedId,
  onSettingsChange,
  onUpdateItem,
  onRemoveItem,
  onClear,
  onExportExcel,
  isExporting,
}: {
  postes: PanierItem[];
  devisSettings: { coefficient: number; coef_final: number };
  highlightedId?: string | null;
  onSettingsChange: (key: "coefficient" | "coef_final", value: number) => void;
  onUpdateItem: (itemId: string, field: string, value: number | boolean) => Promise<void>;
  onRemoveItem: (itemId: string) => void;
  onClear: () => void;
  onExportExcel: () => void;
  isExporting: boolean;
}) {
  const mainPostes   = postes.filter(p => !p.is_option);
  const optionPostes = postes.filter(p => p.is_option);

  // Group postes by ensemble, preserving first-appearance order, "Autre" last
  function groupByEnsemble(items: PanierItem[]): { ensemble: string; rows: PanierItem[] }[] {
    const order: string[] = [];
    const map = new Map<string, PanierItem[]>();
    for (const item of items) {
      const key = item.ensemble?.trim() || "__autre__";
      if (!map.has(key)) { map.set(key, []); order.push(key); }
      map.get(key)!.push(item);
    }
    // "Autre" always last
    const sorted = order.filter(k => k !== "__autre__");
    if (map.has("__autre__")) sorted.push("__autre__");
    return sorted.map(k => ({ ensemble: k === "__autre__" ? "Autre" : k, rows: map.get(k)! }));
  }

  return (
    <aside className="w-[45%] flex-shrink-0 border-l border-border flex flex-col bg-card min-h-0 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border flex-shrink-0">
        <span className="text-sm font-medium">Postes du devis ({mainPostes.length}{optionPostes.length > 0 ? ` + ${optionPostes.length} opt.` : ""})</span>
        <button
          onClick={onClear}
          title="Vider le devis"
          className="text-muted-foreground hover:text-destructive transition-colors"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Settings row — no spinners on these inputs */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-border bg-muted/20 flex-shrink-0">
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground whitespace-nowrap">
          Coeff. fournitures %
          <input
            type="text"
            inputMode="numeric"
            value={devisSettings.coefficient}
            onChange={e => onSettingsChange("coefficient", parseFloat(e.target.value) || 0)}
            className="w-16 text-center text-xs border border-border rounded px-1 py-0.5 bg-transparent focus:outline-none focus:ring-1 focus:ring-foreground/20"
          />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground whitespace-nowrap">
          Coef. final %
          <input
            type="text"
            inputMode="numeric"
            value={devisSettings.coef_final}
            onChange={e => onSettingsChange("coef_final", parseFloat(e.target.value) || 0)}
            className="w-16 text-center text-xs border border-border rounded px-1 py-0.5 bg-transparent focus:outline-none focus:ring-1 focus:ring-foreground/20"
          />
        </label>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <table className="w-full">
          <thead className="sticky top-0 bg-card z-10">
            <tr className="border-b border-border text-xs text-muted-foreground">
              <th className="w-6" />
              <th className="px-2 py-2 text-left font-medium">Poste / Affaire</th>
              <th className="px-1 py-2 text-center font-medium whitespace-nowrap">Étude (j)</th>
              <th className="px-1 py-2 text-center font-medium whitespace-nowrap">Atelier (j)</th>
              <th className="px-1 py-2 text-center font-medium whitespace-nowrap">Client (j)</th>
              <th className="px-1 py-2 text-center font-medium" title="Basculer option / poste">↕</th>
              <th className="w-6" />
            </tr>
          </thead>
          <tbody>
            {groupByEnsemble(mainPostes).map(({ ensemble, rows }) => (
              <Fragment key={`grp-${ensemble}`}>
                <tr>
                  <td colSpan={7} className="px-3 py-1 text-[11px] font-semibold tracking-wide text-muted-foreground bg-muted/40 border-y border-border/40 uppercase">
                    {ensemble}
                  </td>
                </tr>
                {rows.map(item => (
                  <PosteRow
                    key={item.id}
                    item={item}
                    isOption={false}
                    highlighted={highlightedId === item.id}
                    onUpdate={onUpdateItem}
                    onRemove={onRemoveItem}
                  />
                ))}
              </Fragment>
            ))}
            {/* Options section separator */}
            {optionPostes.length > 0 && (
              <tr>
                <td colSpan={7} className="px-3 py-1.5 text-[11px] font-semibold text-amber-600 dark:text-amber-400 bg-amber-500/10 border-y border-amber-500/20">
                  ✦ Options (en sus)
                </td>
              </tr>
            )}
            {groupByEnsemble(optionPostes).map(({ ensemble, rows }) => (
              <Fragment key={`opt-${ensemble}`}>
                {rows.map(item => (
                  <PosteRow
                    key={item.id}
                    item={item}
                    isOption={true}
                    highlighted={highlightedId === item.id}
                    onUpdate={onUpdateItem}
                    onRemove={onRemoveItem}
                  />
                ))}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-border flex-shrink-0">
        <button
          onClick={onExportExcel}
          disabled={isExporting || postes.length === 0}
          className={cn(
            "w-full flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all border",
            isExporting || postes.length === 0
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

// ─── Confidence badge ──────────────────────────────────────────────────────────
const CONFIDENCE_STYLES: Record<string, string> = {
  high:     "bg-green-500/15 border-green-500/30 text-green-700 dark:text-green-400",
  medium:   "bg-yellow-500/15 border-yellow-500/30 text-yellow-700 dark:text-yellow-400",
  low:      "bg-orange-500/15 border-orange-500/30 text-orange-700 dark:text-orange-400",
  fts_only: "bg-muted border-border text-muted-foreground",
};
const CONFIDENCE_LABELS: Record<string, string> = {
  high: "Fort", medium: "Moyen", low: "Faible", fts_only: "FTS",
};

// ─── Candidates Panel ──────────────────────────────────────────────────────────
function CandidatesPanel({
  candidates,
  selectedKeys,
  onToggle,
  onSelectHighMedium,
  onStart,
  onClose,
}: {
  candidates: RfqCandidate[];
  selectedKeys: Set<string>;
  onToggle: (key: string) => void;
  onSelectHighMedium: () => void;
  onStart: () => void;
  onClose: () => void;
}) {
  // Group by dimension
  const byDimension = new Map<string, RfqCandidate[]>();
  for (const c of candidates) {
    const dim = c.dimension || "Général";
    if (!byDimension.has(dim)) byDimension.set(dim, []);
    byDimension.get(dim)!.push(c);
  }
  const maxScore = Math.max(...candidates.map(c => c._score), 1);

  return (
    <div className="border-t border-border bg-card">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">Postes candidats</span>
          <span className="text-xs text-muted-foreground bg-muted rounded-full px-2 py-0.5">
            {candidates.length} postes · {selectedKeys.size} sélectionnés
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onSelectHighMedium}
            className="text-xs text-muted-foreground hover:text-foreground border border-border rounded-full px-2.5 py-1 transition-colors"
          >
            Sélectionner Fort+Moyen
          </button>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Candidates list */}
      <div className="max-h-64 overflow-y-auto divide-y divide-border/40">
        {Array.from(byDimension.entries()).map(([dim, items]) => (
          <div key={dim}>
            <div className="px-4 py-1.5 text-[11px] font-semibold tracking-wide text-muted-foreground bg-muted/40 uppercase">
              {dim}
            </div>
            {items.map((c) => {
              const key = c.nom_poste.toLowerCase();
              const checked = selectedKeys.has(key);
              const scorePct = Math.round((c._score / maxScore) * 100);
              return (
                <label
                  key={key}
                  className="flex items-center gap-3 px-4 py-2 hover:bg-muted/30 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => onToggle(key)}
                    className="w-3.5 h-3.5 accent-foreground shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium truncate">{c.nom_poste}</span>
                      <span className={`shrink-0 text-[10px] rounded-full px-1.5 py-0.5 border ${CONFIDENCE_STYLES[c._confidence] ?? CONFIDENCE_STYLES.fts_only}`}>
                        {CONFIDENCE_LABELS[c._confidence] ?? c._confidence}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      {/* Score bar */}
                      <div className="w-16 h-1 bg-muted rounded-full overflow-hidden shrink-0">
                        <div
                          className="h-full bg-foreground/40 rounded-full"
                          style={{ width: `${scorePct}%` }}
                        />
                      </div>
                      {c.nom_affaire && (
                        <span className="text-[11px] text-muted-foreground truncate">{c.nom_affaire}</span>
                      )}
                    </div>
                  </div>
                  {c.prix_unitaire != null && (
                    <span className="text-[11px] text-muted-foreground shrink-0">{c.prix_unitaire}€</span>
                  )}
                </label>
              );
            })}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-4 py-3 border-t border-border bg-muted/20">
        <span className="text-xs text-muted-foreground">
          {selectedKeys.size === 0 ? "Aucun poste sélectionné" : `${selectedKeys.size} poste(s) à ajouter`}
        </span>
        <button
          onClick={onStart}
          disabled={selectedKeys.size === 0}
          className={cn(
            "flex items-center gap-1.5 rounded-full px-4 py-1.5 text-xs font-medium transition-all",
            selectedKeys.size > 0
              ? "bg-foreground text-background hover:opacity-80"
              : "bg-muted text-muted-foreground cursor-not-allowed"
          )}
        >
          <Send size={12} />
          Démarrer le devis →
        </button>
      </div>
    </div>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────────
export default function DevisPage() {
  const { currentConversationId, createConversation, refreshConversations, mode } = useConversation();
  const { setOpen: setSidebarOpen } = useSidebar();

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
  const [catalogMethod, setCatalogMethod] = useState<"bm25" | "sql">("bm25");

  // RFQ planning progress state
  const [rfqPlanning, setRfqPlanning] = useState<{ status: string; step: string; dimensions_found?: number } | null>(null);

  // RFQ candidates panel
  const [rfqCandidates, setRfqCandidates] = useState<RfqCandidate[]>([]);
  const [selectedCandidateKeys, setSelectedCandidateKeys] = useState<Set<string>>(new Set());
  const [showCandidatesPanel, setShowCandidatesPanel] = useState(false);

  // Tool calls for the current streaming turn
  const [activeToolCalls, setActiveToolCalls] = useState<ToolCallState[]>([]);
  // Map of assistantMessageId → tool calls that belong to it
  const [messageToolCalls, setMessageToolCalls] = useState<Record<string, ToolCallState[]>>({});

  const [postes, setPostes] = useState<PanierItem[]>([]);
  const [devisSettings, setDevisSettings] = useState({ coefficient: 0, coef_final: 0 });
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const [searchWS, setSearchWS] = useState<SearchWorkspaceState | null>(null);
  const highlightTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const settingsSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const activeConvIdRef = useRef<string | null>(null);
  const skipNextReloadRef = useRef<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scopeChoicePendingRef = useRef(false);
  const pendingMessageRef = useRef<string>("");
  // Keep a stable ref to setSidebarOpen to avoid effect re-runs
  const setSidebarOpenRef = useRef(setSidebarOpen);
  setSidebarOpenRef.current = setSidebarOpen;
  const prevPostesLengthRef = useRef(0);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  const sendDepthRef = useRef(0);

  // ── Sidebar auto-close on 0→1 transition, reopen on N→0 ─────────────────
  useEffect(() => {
    const prev = prevPostesLengthRef.current;
    const curr = postes.length;
    prevPostesLengthRef.current = curr;
    if (prev === 0 && curr > 0) setSidebarOpenRef.current(false);
    if (prev > 0 && curr === 0) setSidebarOpenRef.current(true);
  }, [postes.length]);

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

  // ── Load conversation messages + panier + settings when switching ─────────
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
      fetchPanier(currentConversationId).then(setPostes);
      fetchDevisSettings(currentConversationId).then(setDevisSettings);
    } else {
      activeConvIdRef.current = null;
      setMessages([]);
      setPostes([]);
      setDevisSettings({ coefficient: 0, coef_final: 0 });
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

  // ── Settings change (debounced save) ─────────────────────────────────────
  const handleSettingsChange = useCallback((key: "coefficient" | "coef_final", value: number) => {
    setDevisSettings(prev => {
      const next = { ...prev, [key]: value };
      if (settingsSaveTimerRef.current) clearTimeout(settingsSaveTimerRef.current);
      settingsSaveTimerRef.current = setTimeout(async () => {
        const convId = activeConvIdRef.current;
        if (convId) {
          await updateDevisSettings(convId, next.coefficient, next.coef_final).catch(console.error);
        }
      }, 500);
      return next;
    });
  }, []);

  // ── Update panier item (MdO fields) ──────────────────────────────────────
  const handleUpdateItem = useCallback(async (itemId: string, field: string, value: number | boolean) => {
    const convId = activeConvIdRef.current;
    if (!convId) return;

    // For checkbox (is_option): optimistic update
    if (field === "is_option") {
      setPostes(prev => prev.map(i => i.id === itemId ? { ...i, is_option: value as boolean } : i));
    }

    try {
      const updateFields: { nbre_jours_etude?: number; nbre_jours_atelier?: number; nbre_jours_client?: number; is_option?: boolean } = {};
      if (field === "nbre_jours_etude") updateFields.nbre_jours_etude = value as number;
      else if (field === "nbre_jours_atelier") updateFields.nbre_jours_atelier = value as number;
      else if (field === "nbre_jours_client") updateFields.nbre_jours_client = value as number;
      else if (field === "is_option") updateFields.is_option = value as boolean;

      await updatePanierItem(convId, itemId, updateFields);

      // For numeric fields: update confirmed value in state
      if (field !== "is_option") {
        setPostes(prev => prev.map(i => i.id === itemId ? { ...i, [field]: value } : i));
      }
    } catch (err) {
      // For checkbox: revert optimistic update
      if (field === "is_option") {
        setPostes(prev => prev.map(i => i.id === itemId ? { ...i, is_option: !value } : i));
      }
      throw err; // PosteRow handles error for numeric fields
    }
  }, []);

  // ── Add poste helper (shared by choice handlers) ────────────────────────
  const addPosteAndConfirm = useCallback(async (
    convId: string,
    params: { nom_poste: string; nom_affaire?: string; num_poste?: string },
  ): Promise<{ added: boolean; remaining_tasks: { query: string }[] }> => {
    if (params.nom_affaire) await lockDevisAffaire(convId, params.nom_affaire);
    const result = await addPosteToPanierDirect(convId, params);
    if (result.added.length > 0) {
      setPostes((prev) => [...prev, ...result.added]);
      const confirmMsg = { id: crypto.randomUUID(), role: "assistant" as const, content: `**${params.nom_poste}** ajouté au devis.` };
      setMessages((prev) => [...prev, confirmMsg]);
      await addMessage(convId, "assistant", confirmMsg.content);
    }
    return { added: result.added.length > 0, remaining_tasks: result.remaining_tasks ?? [] };
  }, []);

  // ── Send message ──────────────────────────────────────────────────────────
  const handleSend = useCallback(
    async (content?: string, options?: { silent?: boolean }) => {
      const silent = options?.silent ?? false;
      const text = (content ?? input).trim();
      if (!text || isLoading || !llmReady) return;

      // Guard against recursive chains (choice → handleSend → choice → ...)
      if (sendDepthRef.current >= 5) {
        console.warn("handleSend: max recursion depth reached, aborting");
        return;
      }
      sendDepthRef.current++;

      // Intercept: if a scope choice is pending, show choice cards first (skip for silent)
      if (!silent && scopeChoicePendingRef.current) {
        pendingMessageRef.current = text;
        setInput("");
        const choiceId = crypto.randomUUID();
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
      setRfqPlanning(null);
      setShowCandidatesPanel(false);

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
      };
      const assistantId = crypto.randomUUID();

      const history = messagesRef.current
        .filter((m) => m.content && !m.isStreaming)
        .slice(-10)
        .map((m) => ({ role: m.role, content: m.content }));

      if (!silent) setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      setActiveToolCalls([]);
      scrollToBottom();

      // Auto-create conversation on first message
      let convId = activeConvIdRef.current;
      if (messagesRef.current.length === 0) {
        skipNextReloadRef.current = true;
        convId = await createConversation(text.slice(0, 50), mode);
        activeConvIdRef.current = convId;
      }

      let assistantContent = "";
      const turnToolCalls: ToolCallState[] = [];

      try {
        let firstToken = true;

        for await (const event of streamDevisChat(text, collection, convId!, history, catalogMethod)) {
          if (event.error) {
            setError(event.error);
            setIsLoading(false);
            break;
          }

          // RFQ planning progress
          if (event.rfq_planning) {
            if (event.rfq_planning.status === "done") {
              setRfqPlanning(null);
            } else {
              setRfqPlanning(event.rfq_planning);
            }
          }

          // RFQ candidates panel
          if (event.rfq_candidates && event.rfq_candidates.length > 0) {
            setRfqCandidates(event.rfq_candidates);
            // Pre-select high + medium confidence
            const preSelected = new Set(
              event.rfq_candidates
                .filter(c => c._confidence === "high" || c._confidence === "medium")
                .map(c => c.nom_poste.toLowerCase())
            );
            setSelectedCandidateKeys(preSelected);
            setShowCandidatesPanel(true);
          }

          // Catalog preview — show found postes while LLM reasons on specs
          if (event.catalog_preview) {
            const { query, postes, total } = event.catalog_preview;
            const count = total ?? postes.length;
            const posteList = postes.map((p) => p.nom_poste).join(", ");
            const suffix = count > postes.length ? ` (+${count - postes.length} autres)` : "";
            const previewId = crypto.randomUUID();
            setMessages((prev) => [
              ...prev,
              {
                id: previewId,
                role: "assistant" as const,
                content: `🗂️ Catalogue — **${count} poste(s)** trouvé(s) pour « ${query} » : ${posteList}${suffix}`,
                isStreaming: false,
              },
            ]);
          }

          // Docs search result — show which sources were consulted
          if (event.docs_result) {
            const { query, sources, count } = event.docs_result;
            const sourceList = sources.length > 0 ? sources.join(", ") : "aucune source";
            const infoId = crypto.randomUUID();
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

          // Postes update
          if (event.panier) {
            setPostes(event.panier);
          }

          // Settings update from LLM tool call
          if (event.settings) {
            setDevisSettings(event.settings);
          }

          // Highlight the panier item modified by the LLM (auto-clears after 3s)
          if (event.highlight) {
            if (highlightTimerRef.current) clearTimeout(highlightTimerRef.current);
            setHighlightedId(event.highlight);
            highlightTimerRef.current = setTimeout(() => setHighlightedId(null), 3000);
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
                fromSilent: silent,  // track if this card came from a silent continuation
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
            if (event.panier) setPostes(event.panier);
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
        sendDepthRef.current--;
        if (convId) {
          if (!silent) await addMessage(convId, "user", text);
          if (assistantContent) await addMessage(convId, "assistant", assistantContent);
          refreshConversations();
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [input, isLoading, llmReady, collection, catalogMethod, mode, createConversation, refreshConversations, scrollToBottom]
  );

  // ── Choice selection (extracted to useChoiceHandler.ts) ──────────────────
  const handleChoiceSelect = useChoiceHandler({
    messagesRef,
    setMessages,
    setPostes,
    handleSend,
    addPosteAndConfirm,
    activeConvIdRef,
    scopeChoicePendingRef,
    pendingMessageRef,
  });

  // ── Generate devis ────────────────────────────────────────────────────────
  const handleGenerateDevis = useCallback(async () => {
    const convId = activeConvIdRef.current;
    if (!convId || isGenerating) return;

    setIsGenerating(true);
    setError(null);

    const assistantId = crypto.randomUUID();
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

  // ── Postes actions ────────────────────────────────────────────────────────
  const handleRemoveItem = useCallback(
    async (itemId: string) => {
      const convId = activeConvIdRef.current;
      if (!convId) return;
      await removePanierItem(convId, itemId);
      setPostes((prev) => prev.filter((i) => i.id !== itemId));
    },
    []
  );

  const handleClearPanier = useCallback(async () => {
    const convId = activeConvIdRef.current;
    if (!convId) return;
    await clearPanier(convId);
    setPostes([]);
  }, []);

  const handleCandidateToggle = useCallback((key: string) => {
    setSelectedCandidateKeys(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const handleSelectHighMedium = useCallback(() => {
    setSelectedCandidateKeys(
      new Set(
        rfqCandidates
          .filter(c => c._confidence === "high" || c._confidence === "medium")
          .map(c => c.nom_poste.toLowerCase())
      )
    );
  }, [rfqCandidates]);

  const handleCandidatesStart = useCallback(() => {
    const selected = rfqCandidates.filter(c => selectedCandidateKeys.has(c.nom_poste.toLowerCase()));
    if (selected.length === 0) return;
    const lines = selected.map(
      c => `• ${c.nom_poste}${c.nom_affaire ? ` (${c.nom_affaire}` : ""}${c.num_poste ? `, num_poste=${c.num_poste}` : ""}${c.nom_affaire ? ")" : ""}`
    );
    const msg = `[SYSTÈME] Ajoute ces postes au devis :\n${lines.join("\n")}`;
    setShowCandidatesPanel(false);
    handleSend(msg);
  }, [rfqCandidates, selectedCandidateKeys, handleSend]);

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
        <div className="flex items-center gap-2">
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
          <button
            onClick={() => setCatalogMethod(m => m === "bm25" ? "sql" : "bm25")}
            title={catalogMethod === "sql" ? "Mode NL2SQL actif — cliquer pour BM25" : "Mode BM25 actif — cliquer pour NL2SQL"}
            className={cn(
              "text-fluid-xs rounded-full px-2 py-0.5 border transition-colors",
              catalogMethod === "sql"
                ? "bg-violet-500/15 border-violet-500/30 text-violet-600 dark:text-violet-400"
                : "bg-transparent border-border text-muted-foreground hover:text-foreground"
            )}
          >
            {catalogMethod === "sql" ? "NL2SQL" : "BM25"}
          </button>
        </div>
        {!catalogLoaded && (
          <p className="text-fluid-xs text-amber-500/80">
            Catalogue non chargé — lookup désactivé
          </p>
        )}
      </div>
    </>
  );

  return (
    <div className="flex h-screen overflow-hidden">
      {/* ── Main chat column ─────────────────────────────────────────────── */}
      <div className="relative flex flex-col flex-1 min-w-0 overflow-hidden bg-background">
        {/* Header */}
        <header className="flex items-center gap-3 h-14 px-4 border-b border-border flex-shrink-0">
          <SidebarTrigger className="text-muted-foreground hover:text-foreground" />
          <span className="text-sm font-medium text-muted-foreground">Devis de synthèse</span>
        </header>

        {/* LLM loading banner */}
        {!llmReady && !error && (
          <div className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 text-sm text-amber-600 dark:text-amber-400">
            <Spinner className="h-3.5 w-3.5" />
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
                {/* RFQ planning progress banner (first message) */}
                {isLoading && rfqPlanning && (
                  <RfqPlanningBanner planning={rfqPlanning} />
                )}
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
                {isLoading && activeToolCalls.length === 0 && !rfqPlanning && <LoadingDots />}
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

            {showCandidatesPanel && rfqCandidates.length > 0 && (
              <div className="flex-shrink-0 bg-background">
                <div className="max-w-[680px] mx-auto border-x border-border">
                  <CandidatesPanel
                    candidates={rfqCandidates}
                    selectedKeys={selectedCandidateKeys}
                    onToggle={handleCandidateToggle}
                    onSelectHighMedium={handleSelectHighMedium}
                    onStart={handleCandidatesStart}
                    onClose={() => setShowCandidatesPanel(false)}
                  />
                </div>
              </div>
            )}
            <div className="flex-shrink-0 px-4 pb-4 pt-2 bg-background">
              <div className="max-w-[680px] mx-auto">{inputBar}</div>
            </div>
          </>
        )}
      </div>

      {/* ── SearchWorkspace Modal ────────────────────────────────────────── */}
      {searchWS && (
        <SearchWorkspaceModal
          ws={searchWS}
          onUpdate={setSearchWS}
          onClose={() => setSearchWS(null)}
        />
      )}

      {/* ── Devis panel (visible only when postes exist) ──────────────────── */}
      {postes.length > 0 && (
        <DevisPanel
          postes={postes}
          devisSettings={devisSettings}
          highlightedId={highlightedId}
          onSettingsChange={handleSettingsChange}
          onUpdateItem={handleUpdateItem}
          onRemoveItem={handleRemoveItem}
          onClear={handleClearPanier}
          onExportExcel={handleExportExcel}
          isExporting={isExporting}
        />
      )}
    </div>
  );
}
