import type { CatalogElement, Conversation, Message, SSEEvent, DevisSSEEvent, PanierItem } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export async function* streamChat(
  message: string,
  collectionName: string,
  promptName: string = "defaut",
  history: HistoryMessage[] = []
): AsyncGenerator<SSEEvent> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      collection_name: collectionName,
      prompt_name: promptName,
      history,
    }),
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
  limit: number = 50
): Promise<BrowseResult> {
  const response = await fetch(
    `${API_URL}/api/collections/${collectionName}/chunks?offset=${offset}&limit=${limit}`
  );
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
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
  history: { role: "user" | "assistant"; content: string }[]
): AsyncGenerator<DevisSSEEvent> {
  const response = await fetch(`${API_URL}/api/v1/devis/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, collection, conversation_id: conversationId, history }),
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
