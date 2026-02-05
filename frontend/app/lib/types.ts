export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  isStreaming?: boolean;
}

export interface ChatSource {
  fichier: string;
  page: string | number;
  score: number;
}

export interface SSEEvent {
  token?: string;
  sources?: ChatSource[];
  done?: boolean;
  error?: string;
}
