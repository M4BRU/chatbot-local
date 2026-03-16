"use client";

import { useState } from "react";
import {
  debugRAG, fetchCollections, fetchChunks, fetchCollectionSources, debugRfqPlanner,
  type DebugChunk, type DebugResult, type BrowseChunk, type BrowseResult,
  type RfqDebugResult, type RfqDebugFinding, type RfqSqlPoste,
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

function ParentTextBlock({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const lineCount = text.split("\n").length;
  return (
    <div style={{ marginTop: 8, borderLeft: "2px solid #374151", paddingLeft: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1 }}>
          Parent → LLM
        </span>
        <span style={{ fontSize: 11, color: "#4b5563" }}>
          {lineCount} lignes
        </span>
        <button onClick={() => setExpanded(!expanded)} style={{
          background: "none", border: "none", color: "#60a5fa",
          cursor: "pointer", fontSize: 11, padding: 0,
        }}>
          {expanded ? "▲ Réduire" : "▼ Voir"}
        </button>
      </div>
      {expanded && (
        <div style={{
          whiteSpace: "pre-wrap", wordBreak: "break-word",
          fontSize: 12, margin: 0,
          background: "#0d1f0d", padding: "10px 12px", borderRadius: 4,
          color: "#86efac", lineHeight: 1.6,
          maxHeight: 400, overflowY: "auto",
        }}>
          {text}
        </div>
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
        {chunk.parent_text && (
          <span style={{ fontSize: 11, color: "#6b7280", fontStyle: "italic" }}>child</span>
        )}
        <ChunkMeta source={chunk.source} page={chunk.page} chunk_idx={chunk.chunk_idx} machine={chunk.machine} />
      </div>
      <SectionBreadcrumb sections={chunk.sections} />
      <ChunkContent content={chunk.content} preview={chunk.content_preview} />
      {chunk.parent_text && <ParentTextBlock text={chunk.parent_text} />}
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
  const [sources, setSources] = useState<string[]>([]);
  const [selectedSource, setSelectedSource] = useState("");

  async function loadSources(col: string) {
    try {
      const srcs = await fetchCollectionSources(col);
      setSources(srcs);
    } catch {
      setSources([]);
    }
  }

  async function load(col: string, off: number, src: string = selectedSource) {
    setLoading(true); setError(null);
    try {
      setResult(await fetchChunks(col, off, src ? 2000 : PAGE_SIZE, src));
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
    setSelectedSource("");
    setSources([]);
  }

  function handleSourceChange(src: string) {
    setSelectedSource(src);
    setResult(null);
    setOffset(0);
    load(collection, 0, src);
  }

  const totalPages = result ? Math.ceil(result.total / PAGE_SIZE) : 0;
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16, alignItems: "center" }}>
        <select
          value={collection}
          onChange={e => handleCollectionChange(e.target.value)}
          style={selectStyle}
        >
          {collections.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <button
          onClick={async () => { await loadSources(collection); load(collection, 0, ""); }}
          disabled={loading || !collection}
          style={btnStyle(loading)}
        >
          {loading ? "Chargement…" : "Charger les chunks"}
        </button>
        {result && (
          <span style={{ color: "#6b7280", fontSize: 13 }}>
            {result.total} chunks {selectedSource ? `pour ce fichier` : "au total"}
          </span>
        )}
      </div>

      {sources.length > 0 && (
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 16, flexWrap: "wrap" }}>
          <span style={{ fontSize: 13, color: "#6b7280" }}>Filtrer par fichier :</span>
          <select
            value={selectedSource}
            onChange={e => handleSourceChange(e.target.value)}
            style={{ ...selectStyle, maxWidth: 400 }}
          >
            <option value="">— Tous les fichiers —</option>
            {sources.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      )}

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
        {chunk.parent_text && (
          <span style={{ fontSize: 11, color: "#6b7280", fontStyle: "italic" }}>child</span>
        )}
        <ChunkMeta source={chunk.source} page={chunk.page} chunk_idx={chunk.chunk_idx} machine={chunk.machine} />
      </div>
      <SectionBreadcrumb sections={chunk.sections} />
      <ChunkContent content={chunk.content} preview={chunk.content_preview} />
      {chunk.parent_text && <ParentTextBlock text={chunk.parent_text} />}
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

// ── Onglet 3 : RFQ Planner debug ──────────────────────────────────────────────

function SqlPostesBlock({ postes }: { postes: RfqSqlPoste[] }) {
  const [open, setOpen] = useState(false);
  // Dédupliquer par nom_poste
  const unique = postes.filter((p, i, arr) => arr.findIndex(x => x.nom_poste === p.nom_poste) === i);
  return (
    <div style={{ marginBottom: 8 }}>
      <button
        onClick={() => setOpen(!open)}
        style={{ background: "none", border: "none", color: "#f59e0b", fontSize: 12, cursor: "pointer", padding: 0 }}
      >
        {open ? "▲ Masquer" : "▼ Voir"} {unique.length} poste(s) catalogue SQL
      </button>
      {open && (
        <div style={{ marginTop: 6 }}>
          {unique.map((p, i) => (
            <div key={i} style={{
              display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center",
              background: "#1c1a0e", border: "1px solid #78350f",
              borderRadius: 4, padding: "5px 10px", marginBottom: 4, fontSize: 12,
            }}>
              <span style={{ color: "#fcd34d", fontWeight: 600 }}>{p.nom_poste}</span>
              {p.nom_affaire && (
                <span style={{ color: "#92400e", background: "#451a03", borderRadius: 3, padding: "1px 6px" }}>
                  {p.nom_affaire}
                </span>
              )}
              {p.ensemble && <span style={{ color: "#6b7280" }}>{p.ensemble}</span>}
              {p.fournisseur && <span style={{ color: "#9ca3af", fontStyle: "italic" }}>{p.fournisseur}</span>}
              {p.prix_unitaire != null && (
                <span style={{ color: "#34d399", marginLeft: "auto" }}>{p.prix_unitaire}€</span>
              )}
              <span style={{ color: "#374151", fontSize: 11 }}>← {p.composant_nom}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function RfqFindingCard({ finding, isGap }: { finding: RfqDebugFinding; isGap?: boolean }) {
  const [chunksOpen, setChunksOpen] = useState(false);
  const hasComponents = finding.components.length > 0;
  const hasChunks = finding.chunks.length > 0;
  const hasError = !!finding.error;

  return (
    <div style={{
      marginBottom: 16, border: `1px solid ${hasError ? "#7f1d1d" : hasComponents ? "#1e3a5f" : "#374151"}`,
      borderRadius: 8, overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        background: hasError ? "#3b1515" : hasComponents ? "#0f2236" : "#1a1a1a",
        padding: "10px 14px", display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
      }}>
        {isGap && (
          <span style={{ background: "#78350f", color: "#fcd34d", borderRadius: 4, padding: "1px 8px", fontSize: 11, fontWeight: 700 }}>
            GAP
          </span>
        )}
        <span style={{ fontWeight: 700, fontSize: 14, color: "#e5e7eb" }}>{finding.dimension}</span>
        <span style={{ color: "#6b7280", fontSize: 12, marginLeft: "auto" }}>
          {hasComponents
            ? <span style={{ color: "#34d399" }}>✓ {finding.components.length} composant(s)</span>
            : <span style={{ color: "#9ca3af", fontStyle: "italic" }}>0 composant extrait</span>}
          {hasError && <span style={{ color: "#f87171", marginLeft: 8 }}>⚠ erreur</span>}
        </span>
      </div>

      <div style={{ padding: "10px 14px" }}>
        {/* Query */}
        <div style={{ marginBottom: 8 }}>
          <span style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1 }}>Query RAG </span>
          <span style={{ fontSize: 13, color: "#a78bfa", fontFamily: "monospace" }}>{finding.query}</span>
        </div>

        {/* Sources */}
        {finding.sources.length > 0 && (
          <div style={{ marginBottom: 8 }}>
            <span style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1 }}>Sources </span>
            {finding.sources.map((s, i) => (
              <span key={i} style={{ background: "#1f2937", borderRadius: 4, padding: "1px 8px", fontSize: 12, color: "#9ca3af", marginRight: 6 }}>
                {s.split("/").pop()}
              </span>
            ))}
          </div>
        )}

        {/* Composants extraits */}
        {hasComponents && (
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>Composants extraits</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {finding.components.map((c, i) => (
                <span key={i} style={{
                  background: "#0d3320", border: "1px solid #065f46", borderRadius: 4,
                  padding: "2px 10px", fontSize: 12, color: "#34d399",
                }}>
                  {c.nom}{c.specs ? <span style={{ color: "#6b7280", fontSize: 11 }}> — {c.specs}</span> : ""}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Postes catalogue SQL */}
        {finding.sql_postes && finding.sql_postes.length > 0 && (
          <SqlPostesBlock postes={finding.sql_postes} />
        )}
        {finding.sql_postes && finding.sql_postes.length === 0 && (
          <div style={{ fontSize: 12, color: "#6b7280", fontStyle: "italic", marginBottom: 6 }}>
            SQL catalogue : aucun poste trouvé
          </div>
        )}

        {/* Chunks RAG (collapsable) */}
        {hasChunks && (
          <div>
            <button
              onClick={() => setChunksOpen(!chunksOpen)}
              style={{ background: "none", border: "none", color: "#60a5fa", fontSize: 12, cursor: "pointer", padding: 0 }}
            >
              {chunksOpen ? "▲ Masquer" : "▼ Voir"} {finding.chunks.length} chunk(s) RAG bruts
            </button>
            {chunksOpen && (
              <div style={{ marginTop: 8 }}>
                {finding.chunks.map((chunk, i) => (
                  <div key={i} style={{
                    marginBottom: 8, background: "#111", border: "1px solid #1f2937",
                    borderRadius: 4, padding: "8px 10px",
                  }}>
                    <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4 }}>
                      [{i + 1}] <span style={{ color: "#9ca3af" }}>{chunk.source.split("/").pop()}</span>
                    </div>
                    <pre style={{
                      whiteSpace: "pre-wrap", wordBreak: "break-word",
                      fontFamily: "monospace", fontSize: 11, margin: 0, color: "#d1d5db",
                    }}>
                      {chunk.text}
                    </pre>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Erreur */}
        {hasError && (
          <div style={{ color: "#f87171", fontSize: 13, marginTop: 4 }}>⚠ {finding.error}</div>
        )}
      </div>
    </div>
  );
}

function RfqPlannerTab({ collections }: { collections: string[] }) {
  const [message, setMessage] = useState("");
  const [collection, setCollection] = useState(collections[0] || "");
  const [result, setResult] = useState<RfqDebugResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.SyntheticEvent) {
    e.preventDefault();
    if (!message.trim() || !collection) return;
    setLoading(true); setError(null); setResult(null);
    try {
      setResult(await debugRfqPlanner(message, collection));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 24 }}>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <select value={collection} onChange={e => setCollection(e.target.value)} style={selectStyle}>
            {collections.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <button type="submit" disabled={loading || !message.trim() || !collection} style={btnStyle(loading)}>
            {loading ? "Analyse en cours…" : "Lancer le RFQPlanner"}
          </button>
        </div>
        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="Coller le RFQ complet ici…"
          rows={5}
          style={{
            ...inputStyle, width: "100%", resize: "vertical",
            fontFamily: "monospace", fontSize: 13,
          }}
        />
      </form>

      {loading && (
        <div style={{ color: "#60a5fa", fontSize: 14, marginBottom: 16 }}>
          ⏳ Exécution des 4 phases (peut prendre plusieurs minutes selon la collection)…
        </div>
      )}

      {error && <ErrorBox msg={error} />}

      {result && (
        <div>
          {/* Timings */}
          <div style={{
            background: "#1a1a2e", border: "1px solid #2d3a5a",
            borderRadius: 6, padding: "10px 14px", marginBottom: 20,
            display: "flex", gap: 20, flexWrap: "wrap",
          }}>
            <span style={{ fontSize: 12, color: "#6b7280" }}>Timings :</span>
            {Object.entries(result.timing_s).map(([phase, t]) => (
              <span key={phase} style={{ fontSize: 13 }}>
                <span style={{ color: "#6b7280" }}>{phase} </span>
                <span style={{ color: "#fcd34d", fontWeight: 700 }}>{t}s</span>
              </span>
            ))}
          </div>

          {/* Phase 1 : Dimensions */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
              <span style={{ background: "#1e3a5f", color: "#60a5fa", borderRadius: 4, padding: "2px 10px", fontSize: 12, fontWeight: 700 }}>Phase 1</span>
              <span style={{ fontSize: 15, fontWeight: 600 }}>Décomposition RFQ</span>
              <Badge n={result.phases.decomposition.dimensions.length} />
            </div>
            {result.phases.decomposition.error && <ErrorBox msg={result.phases.decomposition.error} />}
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {result.phases.decomposition.dimensions.map((d, i) => (
                <div key={i} style={{
                  background: "#111", border: "1px solid #1f2937",
                  borderRadius: 6, padding: "8px 12px",
                }}>
                  <span style={{ color: "#34d399", fontWeight: 700, fontSize: 13 }}>{d.dimension}</span>
                  <span style={{ color: "#6b7280", fontSize: 12, marginLeft: 10 }}>→ {d.query}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Phase 2 : Recherche */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
              <span style={{ background: "#1e3a5f", color: "#60a5fa", borderRadius: 4, padding: "2px 10px", fontSize: 12, fontWeight: 700 }}>Phase 2</span>
              <span style={{ fontSize: 15, fontWeight: 600 }}>Recherche RAG par dimension</span>
              <Badge n={result.phases.search.findings.length} />
            </div>
            {result.phases.search.findings.map((f, i) => (
              <RfqFindingCard key={i} finding={f} />
            ))}
          </div>

          {/* Phase 3 : Gaps */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
              <span style={{ background: "#1e3a5f", color: "#60a5fa", borderRadius: 4, padding: "2px 10px", fontSize: 12, fontWeight: 700 }}>Phase 3</span>
              <span style={{ fontSize: 15, fontWeight: 600 }}>Gap detection</span>
              <Badge n={result.phases.gaps.detected.length} />
            </div>
            {result.phases.gaps.detected.length === 0 ? (
              <div style={{ color: "#6b7280", fontStyle: "italic", fontSize: 14 }}>Aucun gap critique détecté.</div>
            ) : (
              result.phases.gaps.findings.map((f, i) => (
                <RfqFindingCard key={i} finding={f} isGap />
              ))
            )}
          </div>

          {/* Phase SQL : postes catalogue trouvés */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
              <span style={{ background: "#78350f", color: "#fcd34d", borderRadius: 4, padding: "2px 10px", fontSize: 12, fontWeight: 700 }}>SQL</span>
              <span style={{ fontSize: 15, fontWeight: 600 }}>Postes catalogue trouvés</span>
              <Badge n={[...result.phases.search.findings, ...result.phases.gaps.findings]
                .flatMap(f => f.sql_postes ?? [])
                .filter((p, i, arr) => arr.findIndex(x => x.nom_poste === p.nom_poste) === i).length}
              />
            </div>
            <pre style={{
              whiteSpace: "pre-wrap", wordBreak: "break-word",
              fontFamily: "monospace", fontSize: 12, margin: 0,
              background: "#111", border: "1px solid #78350f",
              borderRadius: 6, padding: "14px 16px", color: "#fcd34d",
              lineHeight: 1.7, maxHeight: 400, overflowY: "auto",
            }}>
              {result.phases.synthesis.structured_context || "(aucun poste trouvé)"}
            </pre>
          </div>

          {/* Phase 4 : Les 2 versions de contexte */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
              <span style={{ background: "#1e3a5f", color: "#60a5fa", borderRadius: 4, padding: "2px 10px", fontSize: 12, fontWeight: 700 }}>Phase 4</span>
              <span style={{ fontSize: 15, fontWeight: 600 }}>Contextes finaux</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {/* Contexte structuré brut */}
              <div>
                <div style={{ fontSize: 12, color: "#9ca3af", marginBottom: 6, display: "flex", gap: 8, alignItems: "center" }}>
                  <span style={{ background: "#1c1a0e", color: "#f59e0b", borderRadius: 3, padding: "1px 8px", fontWeight: 600 }}>BRUT</span>
                  <span>Structuré (sans LLM)</span>
                  <span style={{ color: "#4b5563" }}>{result.phases.synthesis.structured_context?.length ?? 0} chars</span>
                </div>
                <pre style={{
                  whiteSpace: "pre-wrap", wordBreak: "break-word",
                  fontFamily: "monospace", fontSize: 12, margin: 0,
                  background: "#1c1a0e", border: "1px solid #78350f",
                  borderRadius: 6, padding: "12px 14px", color: "#fbbf24",
                  lineHeight: 1.6, maxHeight: 500, overflowY: "auto",
                }}>
                  {result.phases.synthesis.structured_context || "(vide)"}
                </pre>
              </div>
              {/* Contexte LLM synthétisé */}
              <div>
                <div style={{ fontSize: 12, color: "#9ca3af", marginBottom: 6, display: "flex", gap: 8, alignItems: "center" }}>
                  <span style={{ background: "#0f2236", color: "#60a5fa", borderRadius: 3, padding: "1px 8px", fontWeight: 600 }}>LLM</span>
                  <span>Synthèse condensée</span>
                  <span style={{ color: "#4b5563" }}>{result.phases.synthesis.rfq_context.length} chars</span>
                </div>
                <pre style={{
                  whiteSpace: "pre-wrap", wordBreak: "break-word",
                  fontFamily: "monospace", fontSize: 12, margin: 0,
                  background: "#111827", border: "1px solid #1f2937",
                  borderRadius: 6, padding: "12px 14px", color: "#d1d5db",
                  lineHeight: 1.6, maxHeight: 500, overflowY: "auto",
                }}>
                  {result.phases.synthesis.rfq_context || "(vide — aucune information extraite)"}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Page principale ───────────────────────────────────────────────────────────

export default function DebugPage() {
  const [tab, setTab] = useState<"retrieval" | "browse" | "rfq">("retrieval");
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
      <div style={{ display: "flex", gap: 8, marginBottom: 28, flexWrap: "wrap" }}>
        <button style={tabBtnStyle(tab === "retrieval")} onClick={() => setTab("retrieval")}>
          Retrieval (par question)
        </button>
        <button style={tabBtnStyle(tab === "browse")} onClick={() => setTab("browse")}>
          Browse BDD
        </button>
        <button style={tabBtnStyle(tab === "rfq")} onClick={() => setTab("rfq")}>
          RFQ Planner debug
        </button>
      </div>

      {collections.length === 0
        ? <div style={{ color: "#6b7280", fontSize: 14 }}>Chargement des collections…</div>
        : tab === "retrieval"
          ? <RetrievalTab collections={collections} />
          : tab === "rfq"
            ? <RfqPlannerTab collections={collections} />
            : <BrowseTab collections={collections} />
      }
    </div>
  );
}
