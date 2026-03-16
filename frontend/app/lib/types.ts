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
  metrics?: RagMetrics;
  isStreaming?: boolean;
  type?: "text" | "choices";
  choices?: {
    type?: "poste" | "poste_affaire" | "affaire" | "element" | "relevance" | "search_scope" | "search_column" | "findings_confirmation";
    question: string;
    nom_poste?: string;
    components?: string[];
    catalog_matches?: { nom_poste: string; spec_status: "match" | "partial" | "no_match" | "unknown"; note?: string }[];
    doc_only_models?: string[];
    suggestion?: string;
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
  fromSilent?: boolean;  // card generated during a silent LLM call — no chain continuation
}

export interface ChatSource {
  fichier: string;
  page: string | number;
  score: number;
}

export interface RagMetrics {
  chunks_used: number;
  top_score: number | null;
  min_score: number | null;
  retrieval_ms: number;
  pipeline_hash: string;
  search_hash: string;
}

export interface SSEEvent {
  token?: string;
  sources?: ChatSource[];
  metrics?: RagMetrics;
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
  nbre_jours_etude?:   number;
  nbre_jours_atelier?: number;
  nbre_jours_client?:  number;
  is_option?:          boolean;
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
  settings?: { coefficient: number; coef_final: number };
  rfq_planning?: { status: "analyzing" | "searching" | "gap_check" | "synthesizing" | "done"; step: string; dimensions_found?: number };
  choices?: {
    type?: "poste" | "poste_affaire" | "affaire" | "element" | "relevance" | "search_scope" | "search_column" | "findings_confirmation";
    question: string;
    nom_poste?: string;
    components?: string[];
    catalog_matches?: { nom_poste: string; spec_status: "match" | "partial" | "no_match" | "unknown"; note?: string }[];
    doc_only_models?: string[];
    suggestion?: string;
    options: {
      id: string;
      label: string;
      detail?: string;
      description?: string;
      columns?: string[];
      rows?: (string | number | null)[][];
    }[];
  };
  highlight?: string;
  catalog_preview?: {
    query: string;
    postes: { nom_poste: string; ensemble?: string; nom_affaire?: string }[];
  };
}

export type AgentMode = "simple_claude" | "simple_gpt" | "simple_rag" | "combined" | "combined_gpt";
export type AgentStep = "classifying" | "forced" | "claude_query" | "gpt_query" | "rag_search" | "synthesizing";

export interface AgentSSEEvent {
  token?: string;
  agent?: AgentMode;
  agent_step?: AgentStep;
  sources?: ChatSource[];
  done?: boolean;
  error?: string;
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
