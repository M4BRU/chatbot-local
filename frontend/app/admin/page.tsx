"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

// ── Types Excel ───────────────────────────────────────────────────────────────
interface ExcelColumn { name: string; type: string }
interface ExcelTable {
  id: string; filename: string; sheet_name: string; table_name: string;
  columns: ExcelColumn[]; sample_rows: Record<string, unknown>[];
  row_count: number; uploaded_at: string;
}
interface SqlResult { method: string; query_generated: string; results: Record<string, unknown>[]; count: number; error: string | null; llm_answer: string | null }
interface VectorResult { method: string; results: { texte?: string; source?: string; score?: number }[]; count: number; error: string | null; llm_answer: string | null }
interface CompareResponse { question: string; collection: string; sql: SqlResult; vector: VectorResult }

// ── Défi Catalogue ────────────────────────────────────────────────────────────
interface CatalogPipeline { method: string; results: Record<string, unknown>[]; count: number; time_s: number; query_generated?: string; error?: string | null; llm_answer: string | null }
interface CatalogChallengeResponse { question: string; bm25: CatalogPipeline; sql: CatalogPipeline }

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Document {
  nom: string;
  date: string;
  nb_chunks: number;
  nb_pages: number;
}

interface VersionInfo {
  status: "ok" | "stale" | "unknown";
  severity: "required" | "recommended" | "optional" | null;
  changed: string[];
}

const VERSION_BADGE: Record<string, { label: string; className: string }> = {
  required:    { label: "🔴 Ré-indexation requise",     className: "bg-red-100 text-red-700" },
  recommended: { label: "🟠 Mise à jour recommandée",   className: "bg-orange-100 text-orange-700" },
  optional:    { label: "🟡 Mise à jour optionnelle",   className: "bg-yellow-100 text-yellow-700" },
  unknown:     { label: "❓ Version inconnue",           className: "bg-gray-100 text-gray-500" },
};

export default function AdminPage() {
  const [collections, setCollections] = useState<string[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{ current: number; total: number } | null>(null);
  const [message, setMessage] = useState<{ type: "success" | "error" | "warning"; text: string } | null>(null);
  const [catalogStatus, setCatalogStatus] = useState<{ loaded: boolean; rows: number; unique_postes: number } | null>(null);
  const [catalogUploading, setCatalogUploading] = useState(false);
  const [versionStatus, setVersionStatus] = useState<Record<string, VersionInfo>>({});

  // Excel debug
  const [excelTab, setExcelTab] = useState<"tables" | "compare">("compare");
  const [excelTables, setExcelTables] = useState<ExcelTable[]>([]);
  const [compareQuestion, setCompareQuestion] = useState("");
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [comparing, setComparing] = useState(false);
  const [withSynthesis, setWithSynthesis] = useState(false);
  const [selectedFilename, setSelectedFilename] = useState<string>("");

  // Défi Catalogue
  const [challengeQuestion, setChallengeQuestion] = useState("");
  const [challengeResult, setChallengeResult] = useState<CatalogChallengeResponse | null>(null);
  const [challenging, setChallenging] = useState(false);
  const [challengeSynthesis, setChallengeSynthesis] = useState(false);

  // Fetch collections
  const fetchCollections = async () => {
    try {
      const res = await fetch(`${API_URL}/api/collections`);
      const data = await res.json();
      setCollections(data.collections || []);
    } catch {
      setMessage({ type: "error", text: "Erreur de connexion au backend" });
    }
  };

  // Fetch documents for a collection
  const fetchDocuments = async (collectionName: string) => {
    try {
      const res = await fetch(`${API_URL}/api/collections/${collectionName}/documents`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch {
      setDocuments([]);
    }
  };

  const fetchVersionStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/api/collections/version-status`);
      if (!res.ok) return;
      const data: Array<VersionInfo & { name: string }> = await res.json();
      const map: Record<string, VersionInfo> = {};
      for (const item of data) map[item.name] = item;
      setVersionStatus(map);
    } catch { /* silencieux */ }
  };

  const fetchCatalogStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/api/v1/catalog/status`);
      const data = await res.json();
      setCatalogStatus({ loaded: data.loaded, rows: data.rows, unique_postes: data.unique_postes ?? 0 });
    } catch {
      setCatalogStatus(null);
    }
  };

  useEffect(() => {
    fetchCollections();
    fetchCatalogStatus();
    fetchVersionStatus();
  }, []);

  useEffect(() => {
    if (selectedCollection) {
      fetchDocuments(selectedCollection);
      fetchExcelTables(selectedCollection);
      setCompareResult(null);
    }
  }, [selectedCollection]);

  const fetchExcelTables = async (col: string) => {
    try {
      const res = await fetch(`${API_URL}/api/collections/${col}/excel`);
      if (res.ok) { const d = await res.json(); setExcelTables(d.tables || []); }
      else setExcelTables([]);
    } catch { setExcelTables([]); }
  };

  const handleChallenge = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!challengeQuestion.trim()) return;
    setChallenging(true);
    setChallengeResult(null);
    try {
      const res = await fetch(`${API_URL}/api/v1/catalog/challenge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: challengeQuestion, with_synthesis: challengeSynthesis }),
      });
      if (res.ok) setChallengeResult(await res.json());
      else setMessage({ type: "error", text: "Erreur défi catalogue" });
    } catch { setMessage({ type: "error", text: "Erreur de connexion" }); }
    setChallenging(false);
  };

  const handleCompare = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCollection || !compareQuestion.trim()) return;
    setComparing(true);
    setCompareResult(null);
    try {
      const res = await fetch(`${API_URL}/api/collections/${selectedCollection}/excel/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: compareQuestion, with_synthesis: withSynthesis, filename: selectedFilename || null }),
      });
      if (res.ok) setCompareResult(await res.json());
      else setMessage({ type: "error", text: "Erreur compare" });
    } catch { setMessage({ type: "error", text: "Erreur de connexion" }); }
    setComparing(false);
  };

  // Create collection
  const handleCreateCollection = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newCollectionName.trim()) return;

    try {
      const res = await fetch(`${API_URL}/api/collections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newCollectionName.trim() }),
      });

      if (res.ok) {
        setMessage({ type: "success", text: `Collection "${newCollectionName}" créée` });
        setNewCollectionName("");
        fetchCollections();
      } else {
        const data = await res.json();
        setMessage({ type: "error", text: data.detail || "Erreur" });
      }
    } catch {
      setMessage({ type: "error", text: "Erreur de connexion" });
    }
  };

  // Delete collection
  const handleDeleteCollection = async (name: string) => {
    if (!confirm(`Supprimer la collection "${name}" ?`)) return;

    try {
      const res = await fetch(`${API_URL}/api/collections/${name}`, { method: "DELETE" });
      if (res.ok) {
        setMessage({ type: "success", text: `Collection "${name}" supprimée` });
        if (selectedCollection === name) {
          setSelectedCollection(null);
          setDocuments([]);
        }
        fetchCollections();
      }
    } catch {
      setMessage({ type: "error", text: "Erreur lors de la suppression" });
    }
  };

  // Upload documents (multi-fichiers, séquentiel)
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!selectedCollection || !e.target.files?.length) return;

    const files = Array.from(e.target.files);
    setUploading(true);
    setUploadProgress({ current: 0, total: files.length });
    setMessage(null);

    const successes: string[] = [];
    const errors: string[] = [];
    const warnings: string[] = [];

    for (let i = 0; i < files.length; i++) {
      setUploadProgress({ current: i + 1, total: files.length });
      const formData = new FormData();
      formData.append("file", files[i]);

      try {
        const res = await fetch(
          `${API_URL}/api/collections/${selectedCollection}/documents`,
          { method: "POST", body: formData }
        );
        const data = await res.json();
        if (res.ok) {
          successes.push(files[i].name);
          if (data.warnings?.length) {
            warnings.push(`${files[i].name} : ${data.warnings.join(" ")}`);
          }
        } else {
          errors.push(`${files[i].name} : ${data.detail || "erreur"}`);
        }
      } catch {
        errors.push(`${files[i].name} : erreur de connexion`);
      }
    }

    setUploading(false);
    setUploadProgress(null);
    e.target.value = "";
    fetchDocuments(selectedCollection);
    fetchVersionStatus();

    if (errors.length > 0 && successes.length === 0) {
      setMessage({ type: "error", text: `Échec : ${errors.join(", ")}` });
    } else if (warnings.length > 0) {
      const base = `${successes.length} fichier(s) indexé(s)`;
      const errPart = errors.length > 0 ? ` — Erreurs : ${errors.join(", ")}` : "";
      setMessage({ type: "warning", text: `${base}${errPart} — Avertissement : ${warnings.join(" | ")}` });
    } else if (errors.length === 0) {
      setMessage({ type: "success", text: `${successes.length} fichier(s) indexé(s) avec succès` });
    } else {
      setMessage({ type: "success", text: `${successes.length} indexé(s) — Erreurs : ${errors.join(", ")}` });
    }
  };

  // Upload catalog Excel
  const handleCatalogUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setCatalogUploading(true);
    setMessage(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_URL}/api/v1/catalog/upload`, { method: "POST", body: formData });
      const data = await res.json();
      if (res.ok) {
        setMessage({ type: "success", text: `Catalogue chargé : ${data.rows} lignes · ${data.unique_postes ?? 0} postes distincts détectés` });
        fetchCatalogStatus();
      } else {
        setMessage({ type: "error", text: data.detail || "Erreur upload catalogue" });
      }
    } catch {
      setMessage({ type: "error", text: "Erreur de connexion" });
    }
    setCatalogUploading(false);
    e.target.value = "";
  };

  // Delete document
  const handleDeleteDocument = async (docName: string) => {
    if (!selectedCollection || !confirm(`Supprimer "${docName}" ?`)) return;

    try {
      const res = await fetch(
        `${API_URL}/api/collections/${selectedCollection}/documents/${encodeURIComponent(docName)}`,
        { method: "DELETE" }
      );
      if (res.ok) {
        setMessage({ type: "success", text: `"${docName}" supprimé` });
        fetchDocuments(selectedCollection);
      }
    } catch {
      setMessage({ type: "error", text: "Erreur lors de la suppression" });
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-2xl font-bold text-gray-800">Admin - Gestion des documents</h1>
          <div className="flex gap-4 items-center">
            <Link href="/admin/eval" className="text-sm text-purple-600 hover:underline">
              📊 Dashboard éval RAG
            </Link>
            <Link href="/" className="text-blue-600 hover:underline">
              ← Retour au chat
            </Link>
          </div>
        </div>

        {message && (
          <div
            className={`mb-4 p-3 rounded ${
              message.type === "success" ? "bg-green-100 text-green-800" : message.type === "warning" ? "bg-yellow-100 text-yellow-800" : "bg-red-100 text-red-800"
            }`}
          >
            {message.text}
          </div>
        )}

        {/* Catalogue Devis */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-lg font-semibold mb-1">Catalogue — Mode Devis</h2>
          <p className="text-sm text-gray-500 mb-4">
            Fichier Excel (.xlsx/.xls) utilisé pour la recherche de postes et l&apos;enrichissement prix/fournisseur dans le mode Devis.
          </p>
          <div className="flex items-center gap-4">
            <label className="flex-1">
              <input
                type="file"
                accept=".xlsx,.xls"
                onChange={handleCatalogUpload}
                disabled={catalogUploading}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100 disabled:opacity-50"
              />
            </label>
            {catalogStatus && (
              <div className="flex items-center gap-2">
                <span className={`text-sm font-medium px-3 py-1.5 rounded-full ${catalogStatus.loaded ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-500"}`}>
                  {catalogStatus.loaded ? `✓ ${catalogStatus.rows} lignes chargées` : "Non chargé"}
                </span>
                {catalogStatus.loaded && catalogStatus.unique_postes > 0 && (
                  <span className="text-sm font-medium px-3 py-1.5 rounded-full bg-blue-100 text-blue-700">
                    {catalogStatus.unique_postes} postes distincts
                  </span>
                )}
              </div>
            )}
          </div>
          {catalogUploading && (
            <p className="text-sm text-blue-600 mt-2">Chargement en cours…</p>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Collections Panel */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Collections</h2>

            <form onSubmit={handleCreateCollection} className="flex gap-2 mb-4">
              <input
                type="text"
                value={newCollectionName}
                onChange={(e) => setNewCollectionName(e.target.value)}
                placeholder="Nouvelle collection..."
                className="flex-1 px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Créer
              </button>
            </form>

            <ul className="space-y-2">
              {collections.length === 0 ? (
                <li className="text-gray-500 text-sm">Aucune collection</li>
              ) : (
                collections.map((name) => (
                  <li
                    key={name}
                    className={`flex justify-between items-center p-2 rounded cursor-pointer ${
                      selectedCollection === name ? "bg-blue-100" : "hover:bg-gray-100"
                    }`}
                    onClick={() => setSelectedCollection(name)}
                  >
                    <div className="flex flex-col gap-0.5">
                      <span className="text-sm font-medium">{name}</span>
                      {(() => {
                        const v = versionStatus[name];
                        if (!v || v.status === "ok") return null;
                        const badge = VERSION_BADGE[v.severity ?? "unknown"];
                        const title = v.changed.length > 0 ? `Modifié : ${v.changed.join(", ")}` : undefined;
                        return (
                          <span
                            className={`text-xs px-2 py-0.5 rounded-full w-fit ${badge.className}`}
                            title={title}
                          >
                            {badge.label}
                          </span>
                        );
                      })()}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteCollection(name);
                      }}
                      className="text-red-500 hover:text-red-700 text-sm flex-shrink-0"
                    >
                      Supprimer
                    </button>
                  </li>
                ))
              )}
            </ul>
          </div>

          {/* Documents Panel */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">
              Documents {selectedCollection && `- ${selectedCollection}`}
            </h2>

            {selectedCollection ? (
              <>
                <label className="block mb-4">
                  <span className="block text-sm text-gray-600 mb-1">
                    Ajouter des documents (PDF, DOCX, TXT, MD, CSV, XLSX, XLS)
                  </span>
                  <input
                    type="file"
                    accept=".pdf,.txt,.md,.docx,.csv,.xlsx,.xls"
                    multiple
                    onChange={handleUpload}
                    disabled={uploading}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
                  />
                </label>

                {uploading && uploadProgress && (
                  <div className="mb-4">
                    <div className="text-blue-600 text-sm mb-1">
                      Indexation en cours… ({uploadProgress.current}/{uploadProgress.total})
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all"
                        style={{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                <ul className="space-y-2">
                  {documents.length === 0 ? (
                    <li className="text-gray-500 text-sm">Aucun document</li>
                  ) : (
                    documents.map((doc) => (
                      <li
                        key={doc.nom}
                        className="flex justify-between items-center p-2 bg-gray-50 rounded"
                      >
                        <div>
                          <div className="font-medium text-sm">{doc.nom}</div>
                          <div className="text-xs text-gray-500">
                            {doc.nb_pages} pages, {doc.nb_chunks} chunks
                          </div>
                        </div>
                        <button
                          onClick={() => handleDeleteDocument(doc.nom)}
                          className="text-red-500 hover:text-red-700 text-sm"
                        >
                          Supprimer
                        </button>
                      </li>
                    ))
                  )}
                </ul>
              </>
            ) : (
              <p className="text-gray-500 text-sm">
                Sélectionnez une collection pour voir ses documents
              </p>
            )}
          </div>
        </div>
        {/* Excel Debug — visible uniquement si collection sélectionnée */}
        {selectedCollection && (
          <div className="bg-white rounded-lg shadow p-6 mt-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">
                Excel Debug — <span className="text-blue-600">{selectedCollection}</span>
              </h2>
              <div className="flex gap-1 text-sm">
                <button
                  onClick={() => setExcelTab("compare")}
                  className={`px-3 py-1 rounded ${excelTab === "compare" ? "bg-blue-600 text-white" : "bg-gray-100 hover:bg-gray-200"}`}
                >
                  Comparaison pipelines
                </button>
                <button
                  onClick={() => setExcelTab("tables")}
                  className={`px-3 py-1 rounded ${excelTab === "tables" ? "bg-blue-600 text-white" : "bg-gray-100 hover:bg-gray-200"}`}
                >
                  Tables SQL ({excelTables.length})
                </button>
              </div>
            </div>

            {/* ── Onglet Comparaison ─────────────────────────────────────── */}
            {excelTab === "compare" && (
              <div>
                {excelTables.length === 0 && (
                  <p className="text-sm text-gray-400 mb-3">
                    Aucun Excel indexé dans cette collection — uploadez un .xlsx pour activer le pipeline SQL.
                  </p>
                )}
                <form onSubmit={handleCompare} className="flex flex-col gap-2 mb-4">
                  {excelTables.length > 0 && (
                    <select
                      value={selectedFilename}
                      onChange={(e) => { setSelectedFilename(e.target.value); setCompareResult(null); }}
                      className="px-3 py-2 border rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">Tous les fichiers Excel</option>
                      {[...new Set(excelTables.map(t => t.filename))].map(f => (
                        <option key={f} value={f}>{f}</option>
                      ))}
                    </select>
                  )}
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={compareQuestion}
                      onChange={(e) => setCompareQuestion(e.target.value)}
                      placeholder="Ex : quels sont les noms de postes dans la catégorie mécanique ?"
                      className="flex-1 px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                    />
                    <button
                      type="submit"
                      disabled={comparing || !compareQuestion.trim()}
                      className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm"
                    >
                      {comparing ? "…" : "Comparer"}
                    </button>
                  </div>
                  <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={withSynthesis}
                      onChange={(e) => setWithSynthesis(e.target.checked)}
                      className="rounded"
                    />
                    Générer une réponse LLM pour chaque pipeline
                    {withSynthesis && <span className="text-orange-500">(+30–60s)</span>}
                  </label>
                </form>

                {compareResult && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* SQL */}
                    <div className="border rounded p-3">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-semibold text-sm">Pipeline SQL</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                          compareResult.sql.method === "sql" ? "bg-green-100 text-green-700" :
                          compareResult.sql.method === "bm25" ? "bg-yellow-100 text-yellow-700" :
                          "bg-gray-100 text-gray-500"
                        }`}>
                          {compareResult.sql.method} · {compareResult.sql.count} résultat(s)
                        </span>
                      </div>
                      {compareResult.sql.query_generated && (
                        <pre className="text-xs bg-gray-900 text-green-300 rounded p-2 mb-2 overflow-x-auto whitespace-pre-wrap">
                          {compareResult.sql.query_generated}
                        </pre>
                      )}
                      {compareResult.sql.error && (
                        <p className="text-xs text-red-600 mb-2">{compareResult.sql.error}</p>
                      )}
                      <div className="space-y-1 max-h-48 overflow-y-auto">
                        {compareResult.sql.results.map((row, i) => (
                          <pre key={i} className="text-xs bg-gray-50 rounded p-2 overflow-x-auto whitespace-pre-wrap">
                            {JSON.stringify(row, null, 2)}
                          </pre>
                        ))}
                        {compareResult.sql.count === 0 && !compareResult.sql.error && (
                          <p className="text-xs text-gray-400">Aucun résultat SQL</p>
                        )}
                      </div>
                      {compareResult.sql.llm_answer && (
                        <div className="mt-3 border-t pt-2">
                          <p className="text-xs font-medium text-gray-500 mb-1">Réponse LLM</p>
                          <p className="text-sm text-gray-800 whitespace-pre-wrap">{compareResult.sql.llm_answer}</p>
                        </div>
                      )}
                    </div>

                    {/* Vector */}
                    <div className="border rounded p-3">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-semibold text-sm">Pipeline Vecteur</span>
                        <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-purple-100 text-purple-700">
                          vector_rag · {compareResult.vector.count} chunk(s)
                        </span>
                      </div>
                      {compareResult.vector.error && (
                        <p className="text-xs text-red-600 mb-2">{compareResult.vector.error}</p>
                      )}
                      <div className="space-y-2 max-h-48 overflow-y-auto">
                        {compareResult.vector.results.map((chunk, i) => (
                          <div key={i} className="text-xs bg-gray-50 rounded p-2">
                            <div className="flex justify-between text-gray-400 mb-1">
                              <span>{chunk.source || "—"}</span>
                              {chunk.score != null && (
                                <span className={chunk.score >= 0.8 ? "text-green-600" : chunk.score >= 0.6 ? "text-yellow-600" : "text-red-500"}>
                                  score {chunk.score.toFixed(3)}
                                </span>
                              )}
                            </div>
                            <p className="text-gray-700 whitespace-pre-wrap">{chunk.texte}</p>
                          </div>
                        ))}
                        {compareResult.vector.count === 0 && !compareResult.vector.error && (
                          <p className="text-xs text-gray-400">Aucun chunk trouvé</p>
                        )}
                      </div>
                      {compareResult.vector.llm_answer && (
                        <div className="mt-3 border-t pt-2">
                          <p className="text-xs font-medium text-gray-500 mb-1">Réponse LLM</p>
                          <p className="text-sm text-gray-800 whitespace-pre-wrap">{compareResult.vector.llm_answer}</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ── Onglet Tables SQL ──────────────────────────────────────── */}
            {excelTab === "tables" && (
              <div>
                {excelTables.length === 0 ? (
                  <p className="text-sm text-gray-400">Aucun Excel indexé dans cette collection.</p>
                ) : (
                  <div className="space-y-4">
                    {excelTables.map((t) => (
                      <div key={t.id} className="border rounded p-3">
                        <div className="flex items-center justify-between mb-2">
                          <div>
                            <span className="font-medium text-sm">{t.filename}</span>
                            <span className="text-gray-400 text-xs ml-2">feuille : {t.sheet_name}</span>
                          </div>
                          <div className="text-xs text-gray-400">
                            {t.row_count} lignes · table SQL : <code className="bg-gray-100 px-1 rounded">{t.table_name}</code>
                          </div>
                        </div>
                        {/* Colonnes */}
                        <div className="flex flex-wrap gap-1 mb-2">
                          {t.columns.map((c) => (
                            <span key={c.name} className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">
                              {c.name} <span className="text-blue-400">({c.type})</span>
                            </span>
                          ))}
                        </div>
                        {/* Sample rows */}
                        {t.sample_rows.length > 0 && (
                          <div className="overflow-x-auto">
                            <table className="text-xs w-full border-collapse">
                              <thead>
                                <tr className="bg-gray-100">
                                  {t.columns.map((c) => (
                                    <th key={c.name} className="border px-2 py-1 text-left font-medium">{c.name}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {t.sample_rows.map((row, i) => (
                                  <tr key={i} className="hover:bg-gray-50">
                                    {t.columns.map((c) => (
                                      <td key={c.name} className="border px-2 py-1 max-w-xs truncate">
                                        {String(row[c.name] ?? "")}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            <p className="text-xs text-gray-400 mt-1">3 premières lignes (aperçu)</p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Défi Catalogue ───────────────────────────────────────────────── */}
        <div className="bg-white rounded-lg shadow p-6 mt-6">
          <h2 className="text-lg font-semibold mb-4">Défi Catalogue — BM25 vs NL2SQL</h2>
          <form onSubmit={handleChallenge} className="flex flex-col gap-2 mb-4">
            <div className="flex gap-2">
              <input
                type="text"
                value={challengeQuestion}
                onChange={(e) => setChallengeQuestion(e.target.value)}
                placeholder="Ex : quel est le prix du poste Robot Comau ?"
                className="flex-1 px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
              />
              <button
                type="submit"
                disabled={challenging || !challengeQuestion.trim()}
                className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 text-sm"
              >
                {challenging ? "…" : "Lancer le défi"}
              </button>
            </div>
            <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer select-none">
              <input type="checkbox" checked={challengeSynthesis} onChange={(e) => setChallengeSynthesis(e.target.checked)} className="rounded" />
              Générer une réponse LLM pour chaque pipeline
              {challengeSynthesis && <span className="text-orange-500">(+30–60s)</span>}
            </label>
          </form>

          {challengeResult && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* BM25 */}
              <div className="border rounded p-3">
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span className="font-semibold text-sm">Méthode actuelle</span>
                  <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-yellow-100 text-yellow-700">BM25 · {challengeResult.bm25.count} résultat(s)</span>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-500">{challengeResult.bm25.time_s}s</span>
                </div>
                <div className="space-y-1 max-h-48 overflow-y-auto">
                  {challengeResult.bm25.results.map((row, i) => (
                    <pre key={i} className="text-xs bg-gray-50 rounded p-2 overflow-x-auto whitespace-pre-wrap">{JSON.stringify(row, null, 2)}</pre>
                  ))}
                  {challengeResult.bm25.count === 0 && <p className="text-xs text-gray-400">Aucun résultat</p>}
                </div>
                {challengeResult.bm25.llm_answer && (
                  <div className="mt-3 border-t pt-2">
                    <p className="text-xs font-medium text-gray-500 mb-1">Réponse LLM</p>
                    <p className="text-sm text-gray-800 whitespace-pre-wrap">{challengeResult.bm25.llm_answer}</p>
                  </div>
                )}
              </div>

              {/* SQL */}
              <div className="border rounded p-3">
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span className="font-semibold text-sm">Nouvelle méthode</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                    challengeResult.sql.method === "sql" ? "bg-green-100 text-green-700" :
                    challengeResult.sql.method === "bm25" ? "bg-yellow-100 text-yellow-700" :
                    "bg-gray-100 text-gray-500"}`}>
                    {challengeResult.sql.method} · {challengeResult.sql.count} résultat(s)
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-500">{challengeResult.sql.time_s}s</span>
                </div>
                {challengeResult.sql.query_generated && (
                  <pre className="text-xs bg-gray-900 text-green-300 rounded p-2 mb-2 overflow-x-auto whitespace-pre-wrap">{challengeResult.sql.query_generated}</pre>
                )}
                {challengeResult.sql.error && <p className="text-xs text-red-600 mb-2">{challengeResult.sql.error}</p>}
                <div className="space-y-1 max-h-48 overflow-y-auto">
                  {challengeResult.sql.results.map((row, i) => (
                    <pre key={i} className="text-xs bg-gray-50 rounded p-2 overflow-x-auto whitespace-pre-wrap">{JSON.stringify(row, null, 2)}</pre>
                  ))}
                  {challengeResult.sql.count === 0 && !challengeResult.sql.error && <p className="text-xs text-gray-400">Aucun résultat SQL</p>}
                </div>
                {challengeResult.sql.llm_answer && (
                  <div className="mt-3 border-t pt-2">
                    <p className="text-xs font-medium text-gray-500 mb-1">Réponse LLM</p>
                    <p className="text-sm text-gray-800 whitespace-pre-wrap">{challengeResult.sql.llm_answer}</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
