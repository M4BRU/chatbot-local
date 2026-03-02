"use client";

import { useState } from "react";
import {
  debugRAG, fetchCollections, fetchChunks,
  type DebugChunk, type DebugResult, type BrowseChunk, type BrowseResult,
} from "../lib/api";

// ── Composants partagés ───────────────────────────────────────────────────────

function SectionBreadcrumb({ sections }: { sections: string[] }) {
  if (!sections || sections.length === 0) return null;
  return (
    <div style={{ marginTop: 6, fontSize: 12, color: "#9ca3af" }}>
      {sections.map((s, i) => (
        <span key={i}>
          {i > 0 && <span style={{ margin: "0 4px", opacity: 0.5 }}>›</span>}
          <span>{s}</span>
        </span>
      ))}
    </div>
  );
}

function ChunkContent({ content, preview }: { content: string; preview: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={{ marginTop: 8 }}>
      <pre style={{
        whiteSpace: "pre-wrap", wordBreak: "break-word",
        fontFamily: "monospace", fontSize: 12, margin: 0,
        background: "#111", padding: "8px 10px", borderRadius: 4,
        maxHeight: expanded ? "none" : 130, overflow: "hidden",
        color: "#d1d5db",
      }}>
        {expanded ? content : preview}
      </pre>
      {content.length > 300 && (
        <button onClick={() => setExpanded(!expanded)} style={{
          marginTop: 4, background: "none", border: "none",
          color: "#60a5fa", cursor: "pointer", fontSize: 12, padding: 0,
        }}>
          {expanded ? "▲ Réduire" : "▼ Voir tout"}
        </button>
      )}
    </div>
  );
}

function ChunkMeta({ source, page, chunk_idx, machine }: {
  source: string; page: string | number; chunk_idx: number | null; machine: string | null;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
      <span style={{ color: "#aaa", fontSize: 13, fontFamily: "monospace" }}>
        {source}
        {page != null && <> · p.{page}</>}
        {chunk_idx != null && <> · chunk #{chunk_idx}</>}
      </span>
      {machine && (
        <span style={{
          background: "#1e3a5f", color: "#60a5fa",
          borderRadius: 4, padding: "1px 7px", fontSize: 12,
        }}>
          {machine}
        </span>
      )}
    </div>
  );
}

// ── Onglet 1 : Debug retrieval ────────────────────────────────────────────────

function RetrievalTab({ collections }: { collections: string[] }) {
  const [question, setQuestion] = useState("");
  const [collection, setCollection] = useState(collections[0] || "");
  const [result, setResult] = useState<DebugResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.SyntheticEvent) {
    e.preventDefault();
    if (!question.trim() || !collection) return;
    setLoading(true); setError(null); setResult(null);
    try {
      setResult(await debugRAG(question, collection));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <form onSubmit={handleSubmit} style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 24 }}>
        <select
          value={collection}
          onChange={e => setCollection(e.target.value)}
          style={selectStyle}
        >
          {collections.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <input
          type="text" value={question} onChange={e => setQuestion(e.target.value)}
          placeholder="Question à tester…"
          style={{ ...inputStyle, flex: 1, minWidth: 260 }}
        />
        <button type="submit" disabled={loading || !question.trim() || !collection} style={btnStyle(loading)}>
          {loading ? "Recherche…" : "Analyser"}
        </button>
      </form>

      {error && <ErrorBox msg={error} />}

      {result && (
        <>
          {/* Query rewriting */}
          <div style={{
            background: "#1a1a2e", border: "1px solid #2d3a5a",
            borderRadius: 6, padding: "12px 14px", marginBottom: 20,
          }}>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>
              Query rewriting
            </div>
            <div style={{ fontSize: 13, marginBottom: 4 }}>
              <span style={{ color: "#6b7280" }}>Original : </span>
              <span style={{ color: "#d1d5db" }}>{result.original_question}</span>
            </div>
            <div style={{ fontSize: 13 }}>
              <span style={{ color: "#6b7280" }}>Réécrite : </span>
              {result.rewritten_query === result.original_question
                ? <span style={{ color: "#4b5563", fontStyle: "italic" }}>inchangée (pas d'historique)</span>
                : <span style={{ color: "#a78bfa", fontWeight: 600 }}>{result.rewritten_query}</span>}
            </div>
          </div>

          {/* Chunks */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
            <span style={{ fontSize: 15, fontWeight: 600 }}>Chunks récupérés</span>
            <Badge n={result.chunks.length} />
          </div>

          {result.chunks.length === 0
            ? <div style={{ color: "#6b7280", fontStyle: "italic", fontSize: 14 }}>Aucun chunk trouvé.</div>
            : result.chunks.map(chunk => <RetrievalChunkCard key={chunk.rank} chunk={chunk} />)
          }
        </>
      )}
    </div>
  );
}

function RetrievalChunkCard({ chunk }: { chunk: DebugChunk }) {
  const scoreColor = chunk.score >= 0.5 ? "#4ade80" : chunk.score >= 0.2 ? "#facc15" : "#f87171";
  return (
    <div style={cardStyle}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span style={{ background: "#2d2d2d", borderRadius: 4, padding: "2px 8px", fontWeight: 700, fontSize: 13 }}>
          #{chunk.rank}
        </span>
        <span style={{ color: scoreColor, fontWeight: 600, fontSize: 13, fontFamily: "monospace" }}>
          score {chunk.score.toFixed(4)}
        </span>
        <ChunkMeta source={chunk.source} page={chunk.page} chunk_idx={chunk.chunk_idx} machine={chunk.machine} />
      </div>
      <SectionBreadcrumb sections={chunk.sections} />
      <ChunkContent content={chunk.content} preview={chunk.content_preview} />
    </div>
  );
}

// ── Onglet 2 : Browse BDD ─────────────────────────────────────────────────────

const PAGE_SIZE = 50;

function BrowseTab({ collections }: { collections: string[] }) {
  const [collection, setCollection] = useState(collections[0] || "");
  const [result, setResult] = useState<BrowseResult | null>(null);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load(col: string, off: number) {
    setLoading(true); setError(null);
    try {
      setResult(await fetchChunks(col, off, PAGE_SIZE));
      setOffset(off);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  function handleCollectionChange(col: string) {
    setCollection(col);
    setResult(null);
    setOffset(0);
  }

  const totalPages = result ? Math.ceil(result.total / PAGE_SIZE) : 0;
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 24, alignItems: "center" }}>
        <select
          value={collection}
          onChange={e => handleCollectionChange(e.target.value)}
          style={selectStyle}
        >
          {collections.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <button
          onClick={() => load(collection, 0)}
          disabled={loading || !collection}
          style={btnStyle(loading)}
        >
          {loading ? "Chargement…" : "Charger les chunks"}
        </button>
        {result && (
          <span style={{ color: "#6b7280", fontSize: 13 }}>
            {result.total} chunks au total
          </span>
        )}
      </div>

      {error && <ErrorBox msg={error} />}

      {result && (
        <>
          {result.chunks.map((chunk, i) => (
            <BrowseChunkCard key={i} chunk={chunk} globalIndex={offset + i + 1} />
          ))}

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 20 }}>
              <button
                onClick={() => load(collection, offset - PAGE_SIZE)}
                disabled={offset === 0 || loading}
                style={paginBtn}
              >
                ← Précédent
              </button>
              <span style={{ color: "#9ca3af", fontSize: 13 }}>
                Page {currentPage} / {totalPages}
              </span>
              <button
                onClick={() => load(collection, offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= result.total || loading}
                style={paginBtn}
              >
                Suivant →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function BrowseChunkCard({ chunk, globalIndex }: { chunk: BrowseChunk; globalIndex: number }) {
  return (
    <div style={cardStyle}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span style={{ background: "#2d2d2d", borderRadius: 4, padding: "2px 8px", fontWeight: 700, fontSize: 13 }}>
          #{globalIndex}
        </span>
        <ChunkMeta source={chunk.source} page={chunk.page} chunk_idx={chunk.chunk_idx} machine={chunk.machine} />
      </div>
      <SectionBreadcrumb sections={chunk.sections} />
      <ChunkContent content={chunk.content} preview={chunk.content_preview} />
    </div>
  );
}

// ── Styles et utilitaires ─────────────────────────────────────────────────────

const cardStyle: React.CSSProperties = {
  border: "1px solid #333", borderRadius: 6, padding: "12px 14px",
  marginBottom: 10, background: "#1a1a1a",
};

const selectStyle: React.CSSProperties = {
  background: "#1f1f1f", border: "1px solid #333", borderRadius: 6,
  color: "#e5e7eb", padding: "8px 12px", fontSize: 14, minWidth: 180,
};

const inputStyle: React.CSSProperties = {
  background: "#1f1f1f", border: "1px solid #333", borderRadius: 6,
  color: "#e5e7eb", padding: "8px 12px", fontSize: 14,
};

const btnStyle = (disabled: boolean): React.CSSProperties => ({
  background: disabled ? "#374151" : "#2563eb", border: "none",
  borderRadius: 6, color: "#fff", padding: "8px 20px",
  fontSize: 14, fontWeight: 600, cursor: disabled ? "not-allowed" : "pointer",
});

const paginBtn: React.CSSProperties = {
  background: "#1f1f1f", border: "1px solid #333", borderRadius: 6,
  color: "#e5e7eb", padding: "6px 14px", fontSize: 13, cursor: "pointer",
};

function Badge({ n }: { n: number }) {
  return (
    <span style={{ background: "#374151", borderRadius: 12, padding: "2px 10px", fontSize: 13, color: "#9ca3af" }}>
      {n}
    </span>
  );
}

function ErrorBox({ msg }: { msg: string }) {
  return (
    <div style={{
      marginBottom: 16, background: "#3b1515", border: "1px solid #7f1d1d",
      borderRadius: 6, padding: "10px 14px", color: "#fca5a5", fontSize: 14,
    }}>
      {msg}
    </div>
  );
}

// ── Page principale ───────────────────────────────────────────────────────────

export default function DebugPage() {
  const [tab, setTab] = useState<"retrieval" | "browse">("retrieval");
  const [collections, setCollections] = useState<string[]>([]);
  const [collectionsLoaded, setCollectionsLoaded] = useState(false);

  async function ensureCollections() {
    if (collectionsLoaded) return;
    const cols = await fetchCollections();
    setCollections(cols);
    setCollectionsLoaded(true);
  }

  // Charge les collections dès le premier rendu
  if (!collectionsLoaded) ensureCollections();

  const tabBtnStyle = (active: boolean): React.CSSProperties => ({
    background: active ? "#2563eb" : "transparent",
    border: active ? "1px solid #2563eb" : "1px solid #333",
    borderRadius: 6, color: active ? "#fff" : "#9ca3af",
    padding: "7px 18px", fontSize: 14, cursor: "pointer", fontWeight: active ? 600 : 400,
  });

  return (
    <div style={{
      minHeight: "100vh", background: "#0f0f0f", color: "#e5e7eb",
      fontFamily: "system-ui, sans-serif", padding: "32px 24px",
      maxWidth: 960, margin: "0 auto",
    }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>Debug RAG</h1>
      <p style={{ color: "#6b7280", fontSize: 14, marginBottom: 24 }}>
        Explore le pipeline de retrieval et le contenu de la base vectorielle.
      </p>

      {/* Onglets */}
      <div style={{ display: "flex", gap: 8, marginBottom: 28 }}>
        <button style={tabBtnStyle(tab === "retrieval")} onClick={() => setTab("retrieval")}>
          Retrieval (par question)
        </button>
        <button style={tabBtnStyle(tab === "browse")} onClick={() => setTab("browse")}>
          Browse BDD
        </button>
      </div>

      {collections.length === 0
        ? <div style={{ color: "#6b7280", fontSize: 14 }}>Chargement des collections…</div>
        : tab === "retrieval"
          ? <RetrievalTab collections={collections} />
          : <BrowseTab collections={collections} />
      }
    </div>
  );
}
