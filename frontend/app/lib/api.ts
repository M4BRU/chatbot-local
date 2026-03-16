import type { AgentMode, AgentSSEEvent, CatalogElement, Conversation, Message, SSEEvent, DevisSSEEvent, PanierItem } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export async function* streamChat(
  message: string,
  collectionName: string,
  promptName: string = "defaut",
  history: HistoryMessage[] = [],
  convId?: string,
): AsyncGenerator<SSEEvent> {
  const body: Record<string, unknown> = {
    message,
    collection_name: collectionName,
    prompt_name: promptName,
    history,
  };
  if (convId) body.conv_id = convId;

  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6)) as SSEEvent;
          yield data;
        } catch {
          // Ignore parse errors
        }
      }
    }
  }
}

export async function* streamAgentChat(
  message: string,
  collectionName: string,
  forceMode?: AgentMode
): AsyncGenerator<AgentSSEEvent> {
  const body: Record<string, unknown> = { message, collection_name: collectionName };
  if (forceMode) body.force_mode = forceMode;

  const response = await fetch(`${API_URL}/api/v1/agent/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6)) as AgentSSEEvent;
        } catch { /* ignore */ }
      }
    }
  }
}

export async function fetchLLMStatus(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/api/v1/status`);
    if (!response.ok) return false;
    const data = await response.json();
    return data.llm_ready === true;
  } catch {
    return false;
  }
}

export async function fetchCollections(): Promise<string[]> {
  const response = await fetch(`${API_URL}/api/collections`);
  if (!response.ok) return [];
  const data = await response.json();
  return data.collections || [];
}

export interface DebugChunk {
  rank: number;
  score: number;
  source: string;
  page: string | number;
  chunk_idx: number | null;
  machine: string | null;
  sections: string[];
  content: string;
  content_preview: string;
  parent_text?: string;
}

export interface DebugResult {
  original_question: string;
  rewritten_query: string;
  chunks: DebugChunk[];
}

export interface BrowseChunk {
  source: string;
  page: string | number;
  chunk_idx: number | null;
  machine: string | null;
  sections: string[];
  content: string;
  content_preview: string;
  parent_text?: string;
}

export interface BrowseResult {
  total: number;
  offset: number;
  limit: number;
  chunks: BrowseChunk[];
}

export async function fetchChunks(
  collectionName: string,
  offset: number = 0,
  limit: number = 50,
  source: string = ""
): Promise<BrowseResult> {
  const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
  if (source) params.set("source", source);
  const response = await fetch(`${API_URL}/api/collections/${collectionName}/chunks?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

export async function fetchCollectionSources(collectionName: string): Promise<string[]> {
  const response = await fetch(`${API_URL}/api/collections/${collectionName}/sources`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  return data.sources as string[];
}

// ─── Conversation API ────────────────────────────────────────────────────────

export async function fetchConversations(): Promise<Conversation[]> {
  const response = await fetch(`${API_URL}/api/conversations`);
  if (!response.ok) return [];
  return response.json();
}

export async function createConversation(title: string, mode: string): Promise<Conversation> {
  const response = await fetch(`${API_URL}/api/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, mode }),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

export async function getConversationMessages(id: string): Promise<Message[]> {
  const response = await fetch(`${API_URL}/api/conversations/${id}/messages`);
  if (!response.ok) return [];
  return response.json();
}

export async function addMessage(conversationId: string, role: string, content: string): Promise<void> {
  await fetch(`${API_URL}/api/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, content }),
  });
}

export async function updateConversationTitle(id: string, title: string): Promise<void> {
  await fetch(`${API_URL}/api/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export async function deleteConversation(id: string): Promise<void> {
  await fetch(`${API_URL}/api/conversations/${id}`, { method: "DELETE" });
}

// ─── Devis API ───────────────────────────────────────────────────────────────

export async function* streamDevisChat(
  message: string,
  collection: string,
  conversationId: string,
  history: { role: "user" | "assistant"; content: string }[],
  catalogMethod: string = "bm25"
): AsyncGenerator<DevisSSEEvent> {
  const response = await fetch(`${API_URL}/api/v1/devis/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      collection,
      conversation_id: conversationId,
      history,
      catalog_method: catalogMethod,
    }),
  });
  if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6)) as DevisSSEEvent;
        } catch { /* ignore */ }
      }
    }
  }
}

export async function* streamGenerateDevis(
  conversationId: string
): AsyncGenerator<DevisSSEEvent> {
  const response = await fetch(`${API_URL}/api/v1/devis/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId }),
  });
  if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6)) as DevisSSEEvent;
        } catch { /* ignore */ }
      }
    }
  }
}

export async function fetchPanier(conversationId: string): Promise<PanierItem[]> {
  const response = await fetch(`${API_URL}/api/v1/devis/${conversationId}/panier`);
  if (!response.ok) return [];
  return response.json();
}

export async function clearPanier(conversationId: string): Promise<void> {
  await fetch(`${API_URL}/api/v1/devis/${conversationId}/panier`, { method: "DELETE" });
}

export async function removePanierItem(conversationId: string, itemId: string): Promise<void> {
  await fetch(`${API_URL}/api/v1/devis/${conversationId}/panier/${itemId}`, { method: "DELETE" });
}

export async function fetchPosteElements(nomPoste: string, nomAffaire?: string): Promise<CatalogElement[]> {
  let url = `${API_URL}/api/v1/catalog/elements?nom_poste=${encodeURIComponent(nomPoste)}`;
  if (nomAffaire) url += `&nom_affaire=${encodeURIComponent(nomAffaire)}`;
  const res = await fetch(url);
  if (!res.ok) return [];
  return res.json();
}

export async function addPosteToPanierDirect(
  conversationId: string,
  posteData: { nom_poste: string; nom_affaire?: string; num_poste?: string; quantite?: number }
): Promise<{ added: PanierItem[]; remaining_tasks: { query: string }[] }> {
  const res = await fetch(`${API_URL}/api/v1/devis/${conversationId}/panier/poste`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(posteData),
  });
  if (!res.ok) throw new Error(await res.text());
  const result = await res.json();
  return { added: result.added || [], remaining_tasks: result.remaining_tasks || [] };
}

export async function addElementToPanierDirect(
  conversationId: string,
  elementData: Record<string, string>
): Promise<PanierItem[]> {
  const res = await fetch(`${API_URL}/api/v1/devis/${conversationId}/panier/element`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(elementData),
  });
  if (!res.ok) throw new Error(await res.text());
  const result = await res.json();
  return result.added || [];
}

export async function lockDevisAffaire(conversationId: string, nomAffaire: string): Promise<void> {
  await fetch(`${API_URL}/api/v1/devis/${conversationId}/lock-affaire`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nom_affaire: nomAffaire }),
  });
}

export async function setSearchScope(conversationId: string, searchAll: boolean): Promise<void> {
  await fetch(`${API_URL}/api/v1/devis/${conversationId}/search-scope`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ search_all: searchAll }),
  });
}

export async function fetchCatalogStatus(): Promise<{ loaded: boolean; rows: number; columns: string[] }> {
  const response = await fetch(`${API_URL}/api/v1/catalog/status`);
  if (!response.ok) return { loaded: false, rows: 0, columns: [] };
  return response.json();
}

export async function fetchDevisSettings(conversationId: string): Promise<{ coefficient: number; coef_final: number }> {
  const res = await fetch(`${API_URL}/api/v1/devis/${conversationId}/settings`);
  if (!res.ok) return { coefficient: 0, coef_final: 0 };
  return res.json();
}

export async function updateDevisSettings(conversationId: string, coefficient: number, coefFinal: number): Promise<void> {
  await fetch(`${API_URL}/api/v1/devis/${conversationId}/settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ coefficient, coef_final: coefFinal }),
  });
}

export async function updatePanierItem(
  conversationId: string,
  itemId: string,
  fields: { nbre_jours_etude?: number; nbre_jours_atelier?: number; nbre_jours_client?: number; is_option?: boolean }
): Promise<PanierItem> {
  const res = await fetch(`${API_URL}/api/v1/devis/${conversationId}/panier/${itemId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(fields),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function exportDevisExcel(conversationId: string): Promise<void> {
  const response = await fetch(`${API_URL}/api/v1/devis/${conversationId}/export`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `devis_${conversationId.slice(0, 8)}.xlsx`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ─── Debug RAG ───────────────────────────────────────────────────────────────

export interface RfqDebugChunk {
  source: string;
  text: string;
}

export interface RfqDebugComponent {
  nom: string;
  specs?: string;
}

export interface RfqSqlPoste {
  composant_nom: string;
  nom_poste: string;
  nom_affaire?: string;
  ensemble?: string;
  fournisseur?: string;
  prix_unitaire?: number;
}

export interface RfqDebugFinding {
  dimension: string;
  query: string;
  chunks: RfqDebugChunk[];
  components: RfqDebugComponent[];
  sources: string[];
  sql_postes?: RfqSqlPoste[];
  error?: string;
}

export interface RfqDebugResult {
  phases: {
    decomposition: {
      dimensions: Array<{ dimension: string; query: string }>;
      error?: string;
    };
    search: {
      findings: RfqDebugFinding[];
    };
    gaps: {
      detected: Array<{ dimension: string; query: string }>;
      findings: RfqDebugFinding[];
      error?: string;
    };
    synthesis: {
      rfq_context: string;
      structured_context: string;
    };
  };
  timing_s: Record<string, number>;
}

export async function debugRfqPlanner(
  message: string,
  collection: string,
): Promise<RfqDebugResult> {
  const response = await fetch(`${API_URL}/api/v1/devis/rfq-debug`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, collection }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

export async function debugRAG(
  message: string,
  collectionName: string,
  promptName: string = "defaut"
): Promise<DebugResult> {
  const response = await fetch(`${API_URL}/api/chat/debug`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      collection_name: collectionName,
      prompt_name: promptName,
      history: [],
    }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return response.json();
}
