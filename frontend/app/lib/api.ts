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

export async function fetchCollections(): Promise<string[]> {
  const response = await fetch(`${API_URL}/api/collections`);
  if (!response.ok) return [];
  const data = await response.json();
  return data.collections || [];
}
