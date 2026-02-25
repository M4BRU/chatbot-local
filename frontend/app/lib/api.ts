import type { SSEEvent } from "./types";

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
