import { useCallback } from "react";
import {
  addElementToPanierDirect,
  addMessage,
  lockDevisAffaire,
  setSearchScope,
} from "@/app/lib/api";
import type { ChatMessage, PanierItem } from "@/app/lib/types";

// ─── Types ───────────────────────────────────────────────────────────────────
type SetMessages = React.Dispatch<React.SetStateAction<ChatMessage[]>>;
type SetPostes = React.Dispatch<React.SetStateAction<PanierItem[]>>;
type HandleSend = (content?: string, options?: { silent?: boolean }) => void | Promise<void>;
type AddPosteAndConfirm = (
  convId: string,
  params: { nom_poste: string; nom_affaire?: string; num_poste?: string },
) => Promise<{ added: boolean; remaining_tasks: { query: string }[] }>;

interface Deps {
  messagesRef: React.RefObject<ChatMessage[]>;
  setMessages: SetMessages;
  setPostes: SetPostes;
  handleSend: HandleSend;
  addPosteAndConfirm: AddPosteAndConfirm;
  activeConvIdRef: React.RefObject<string | null>;
  scopeChoicePendingRef: React.MutableRefObject<boolean>;
  pendingMessageRef: React.MutableRefObject<string>;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Show affaire sub-choice cards when a poste exists in multiple affaires */
function showAffaireSubChoices(
  setMessages: SetMessages,
  nomPoste: string,
  uniqueAffaires: Record<string, string>[],
  question?: string,
) {
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
  const subId = crypto.randomUUID();
  const subQ = question ?? `Le poste « ${nomPoste} » existe dans plusieurs affaires. Quelle affaire utiliser ?`;
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

/** Deduplicate occurrences by nom_affaire */
function dedupeAffaires(occurrences: Record<string, string>[]) {
  return occurrences.filter(
    (occ, idx) => occurrences.findIndex((o) => o.nom_affaire === occ.nom_affaire) === idx,
  );
}

/** Add poste (single affaire) and chain to next task if any */
async function addPosteOrShowAffaires(
  deps: Pick<Deps, "setMessages" | "addPosteAndConfirm" | "handleSend">,
  convId: string,
  nomPoste: string,
  occurrences: Record<string, string>[],
  opts?: { chain?: boolean; question?: string },
) {
  const uniqueAffaires = dedupeAffaires(occurrences);
  if (uniqueAffaires.length <= 1) {
    const occ = uniqueAffaires[0] ?? {};
    const { remaining_tasks } = await deps.addPosteAndConfirm(convId, {
      nom_poste: nomPoste,
      nom_affaire: occ.nom_affaire,
      num_poste: occ.num_poste,
    });
    if (opts?.chain && remaining_tasks.length > 0) {
      const next = remaining_tasks[0].query;
      deps.handleSend(`Cherche "${next}". Lance search_catalog(query="${next}", column="nom_poste").`, { silent: true });
    }
  } else {
    showAffaireSubChoices(deps.setMessages, nomPoste, uniqueAffaires, opts?.question);
  }
}

// ─── Choice handlers by type ─────────────────────────────────────────────────

async function handleElement(
  deps: Pick<Deps, "setMessages" | "setPostes" | "addPosteAndConfirm">,
  convId: string,
  id: string,
  label: string,
) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const parsed: Record<string, any> = JSON.parse(id);
  const action: string = parsed.action ?? "";

  if (action === "add_poste") {
    await deps.addPosteAndConfirm(convId, {
      nom_poste: parsed.nom_poste,
      nom_affaire: parsed.nom_affaire,
      num_poste: parsed.num_poste,
    });
  } else if (action === "show_elements") {
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
    const subId = crypto.randomUUID();
    const subQ = "Lequel des éléments souhaitez-vous ajouter au devis ?";
    deps.setMessages((prev) => [
      ...prev,
      {
        id: subId, role: "assistant" as const, content: subQ,
        type: "choices" as const,
        choices: { type: "element" as const, question: subQ, options: subOptions },
        choiceSelected: false,
      },
    ]);
  } else if (action === "show_element_affaires") {
    const occs: Record<string, string>[] = parsed.occurrences ?? [];
    const elText: string = parsed.element_text ?? label;
    const subOptions = occs.map((occ) => ({
      id: JSON.stringify({ action: "add_element", ...occ }),
      label: occ.nom_affaire || "?",
      detail: `Poste : ${occ.nom_poste}${occ.fournisseur ? ` · ${occ.fournisseur}` : ""}`,
    }));
    const subId = crypto.randomUUID();
    const subQ = `L'élément « ${elText} » est disponible dans plusieurs affaires. Laquelle utiliser ?`;
    deps.setMessages((prev) => [
      ...prev,
      {
        id: subId, role: "assistant" as const, content: subQ,
        type: "choices" as const,
        choices: { type: "element" as const, question: subQ, options: subOptions },
        choiceSelected: false,
      },
    ]);
  } else if (action === "add_element") {
    const newItems = await addElementToPanierDirect(convId, parsed as Record<string, string>);
    if (newItems.length > 0) {
      deps.setPostes((prev) => [...prev, ...newItems]);
      const confirmMsg = { id: crypto.randomUUID(), role: "assistant" as const, content: `**${parsed.elements ?? label}** ajouté au devis.` };
      deps.setMessages((prev) => [...prev, confirmMsg]);
      await addMessage(convId, "assistant", confirmMsg.content);
    }
  }
}

async function handleRelevance(
  deps: Pick<Deps, "setMessages" | "handleSend" | "addPosteAndConfirm">,
  convId: string | null,
  id: string,
  label: string,
) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const parsed: Record<string, any> = JSON.parse(id);

  if (parsed.action === "search_docs") {
    deps.handleSend(
      `Aucun résultat pertinent dans le catalogue pour « ${parsed.query} ». Cherche dans la documentation PDF avec les termes de la description technique et des composants.`,
    );
  } else if (parsed.action === "refine") {
    // Dismiss — user will retype
  } else if (parsed.nom_poste) {
    if (!convId) return;
    await addPosteOrShowAffaires(deps, convId, parsed.nom_poste, parsed.occurrences ?? []);
  }
}

async function handlePoste(
  deps: Pick<Deps, "setMessages" | "handleSend" | "addPosteAndConfirm">,
  convId: string,
  id: string,
  label: string,
) {
  const parsed: { nom_poste: string; occurrences: Array<Record<string, string>> } = JSON.parse(id);
  await addPosteOrShowAffaires(
    deps,
    convId,
    parsed.nom_poste,
    parsed.occurrences ?? [],
    { chain: true, question: `Le poste « ${parsed.nom_poste} » existe dans plusieurs affaires. Quelle affaire utiliser pour ce devis ?` },
  );
}

async function handlePosteAffaire(
  deps: Pick<Deps, "addPosteAndConfirm" | "handleSend">,
  convId: string,
  id: string,
) {
  const parsed: { nom_poste: string; nom_affaire: string; num_poste: string } = JSON.parse(id);
  const { remaining_tasks } = await deps.addPosteAndConfirm(convId, {
    nom_poste: parsed.nom_poste,
    nom_affaire: parsed.nom_affaire,
    num_poste: parsed.num_poste,
  });
  if (remaining_tasks.length > 0) {
    const next = remaining_tasks[0].query;
    deps.handleSend(`Cherche "${next}". Lance search_catalog(query="${next}", column="nom_poste").`, { silent: true });
  }
}

function handleSearchColumn(deps: Pick<Deps, "handleSend">, id: string) {
  const parsed: { action: string; column?: string; query?: string } = JSON.parse(id);
  if (parsed.action === "search_column" && parsed.column && parsed.query) {
    const colLabels: Record<string, string> = {
      elements: "sous-composants (éléments)",
      ensemble: "type d'ensemble",
      nom_affaire: "nom d'affaire",
      fournisseur: "fournisseur",
    };
    const colLabel = colLabels[parsed.column] ?? parsed.column;
    deps.handleSend(
      `Cherche "${parsed.query}" dans la colonne ${colLabel}. Appelle search_catalog avec column="${parsed.column}".`,
    );
  }
}

function handleFindingsConfirmation(deps: Pick<Deps, "handleSend">, id: string) {
  const parsed: { action: string; components?: string[] } = JSON.parse(id);
  if (parsed.action === "confirm_findings" && parsed.components?.length) {
    const componentList = parsed.components.map((c) => `"${c}"`).join(", ");
    deps.handleSend(
      `[SYSTÈME] Composants validés : ${componentList}. Cherche maintenant chacun dans le catalogue avec search_catalog(query=..., column="nom_poste").`,
      { silent: true },
    );
  }
}

async function handleSearchScope(
  deps: Pick<Deps, "handleSend" | "scopeChoicePendingRef" | "pendingMessageRef">,
  convId: string,
  id: string,
) {
  const parsed: { action: string } = JSON.parse(id);
  if (parsed.action === "search_all_affaires") {
    await setSearchScope(convId, true);
  }
  deps.scopeChoicePendingRef.current = false;
  const savedMsg = deps.pendingMessageRef.current;
  deps.pendingMessageRef.current = "";
  if (savedMsg) deps.handleSend(savedMsg);
}

function handleAffaireFallback(
  deps: Pick<Deps, "handleSend">,
  convId: string | null,
  id: string,
  label: string,
  nomPoste?: string,
) {
  if (convId) {
    lockDevisAffaire(convId, id).catch(console.error);
  }
  const userMsg = nomPoste
    ? `Affaire sélectionnée : "${label}". Ajoute le poste "${nomPoste}" au devis.`
    : label;
  deps.handleSend(userMsg);
}

// ─── Hook ────────────────────────────────────────────────────────────────────
export function useChoiceHandler(deps: Deps) {
  return useCallback(
    async (id: string, label: string, messageId: string) => {
      const { messagesRef, setMessages, activeConvIdRef } = deps;

      // Disable the choice cards
      setMessages((prev) =>
        prev.map((m) => (m.id === messageId ? { ...m, choiceSelected: true } : m)),
      );

      const choiceMsg = messagesRef.current.find((m) => m.id === messageId);
      const choiceType = choiceMsg?.choices?.type;
      const nomPoste = choiceMsg?.choices?.nom_poste;
      const convId = activeConvIdRef.current;

      try {
        if (choiceType === "element") {
          if (!convId) return;
          await handleElement(deps, convId, id, label);
        } else if (choiceType === "relevance") {
          await handleRelevance(deps, convId, id, label);
        } else if (choiceType === "poste") {
          if (!convId) return;
          await handlePoste(deps, convId, id, label);
        } else if (choiceType === "poste_affaire") {
          if (!convId) return;
          await handlePosteAffaire(deps, convId, id);
        } else if (choiceType === "search_column") {
          handleSearchColumn(deps, id);
        } else if (choiceType === "findings_confirmation") {
          handleFindingsConfirmation(deps, id);
        } else if (choiceType === "search_scope") {
          if (!convId) return;
          await handleSearchScope(deps, convId, id);
        } else {
          handleAffaireFallback(deps, convId, id, label, nomPoste);
        }
      } catch (err) {
        console.error("Choice handler error:", err);
        // Fallback for JSON parse failures on poste/relevance
        if (choiceType === "poste") {
          deps.handleSend(`Poste sélectionné : "${label}". Recherche ce poste exact et ajoute-le au devis.`);
        } else if (choiceType === "relevance") {
          deps.handleSend(label);
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [deps.handleSend],
  );
}
