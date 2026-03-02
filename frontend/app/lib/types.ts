export interface Conversation {
  id: string;
  title: string;
  mode: string;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: string;
  content: string;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  isStreaming?: boolean;
  type?: "text" | "choices";
  choices?: {
    type?: "poste" | "poste_affaire" | "affaire" | "element" | "relevance" | "search_scope" | "search_column";
    question: string;
    nom_poste?: string;
    options: {
      id: string;
      label: string;
      detail?: string;
      description?: string;
      columns?: string[];
      rows?: (string | number | null)[][];
    }[];
  };
  choiceSelected?: boolean;
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

export interface PanierItem {
  id: string;
  conversation_id: string;
  nom_poste: string;
  ensemble?: string;
  quantite: number;
  fournisseur?: string;
  fourniture?: string;
  num_affaire?: string;
  nom_affaire?: string;
  item_type?: "poste" | "element";
  created_at: string;
}

export interface DevisSSEEvent {
  token?: string;
  tool_call?: { name: string; status: "running" | "done" };
  docs_result?: { query: string; sources: string[]; count: number };
  panier?: PanierItem[];
  done?: boolean;
  ask_scope?: boolean;
  error?: string;
  choices?: {
    type?: "poste" | "poste_affaire" | "affaire" | "element" | "relevance" | "search_scope" | "search_column";
    question: string;
    nom_poste?: string;
    options: {
      id: string;
      label: string;
      detail?: string;
      description?: string;
      columns?: string[];
      rows?: (string | number | null)[][];
    }[];
  };
}

export interface CatalogElement {
  nom_poste?: string;
  elements?: string;
  fournisseur?: string;
  fourniture?: string | number;
  num_poste?: string;
  num_ensemble?: string;
  ensemble?: string;
  nom_affaire?: string;
  num_affaire?: string;
}
