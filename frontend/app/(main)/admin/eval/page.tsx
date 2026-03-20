"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const apiFetch = (input: RequestInfo | URL, init?: RequestInit) =>
  fetch(input, { credentials: "include", ...init });

interface EvalStats {
  total: number;
  avg_faithfulness: number | null;
  avg_answer_relevancy: number | null;
  avg_context_precision: number | null;
  avg_retrieval_ms: number | null;
  avg_eval_ms: number | null;
  errors: number;
  pending?: number;
}

interface TrendDay {
  day: string;
  count: number;
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
}

interface HashGroup {
  pipeline_hash: string;
  search_hash: string;
  count: number;
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
  first_seen: string;
  last_seen: string;
}

interface EvalRow {
  id: number;
  timestamp: string;
  collection: string;
  pipeline_hash: string;
  search_hash: string;
  question: string;
  answer_preview: string;
  context_preview: string;
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
  retrieval_ms: number | null;
  eval_ms: number | null;
  eval_model: string;
  eval_error: string | null;
}

function fmt(v: number | null, decimals = 2): string {
  if (v == null) return "—";
  return v.toFixed(decimals);
}

function ScoreBadge({ value }: { value: number | null }) {
  if (value == null) return <span className="text-gray-400">—</span>;
  const pct = Math.round(value * 100);
  const color =
    pct >= 80 ? "text-green-700 bg-green-50" :
    pct >= 60 ? "text-yellow-700 bg-yellow-50" :
    "text-red-700 bg-red-50";
  return (
    <span className={`inline-block text-xs font-semibold px-2 py-0.5 rounded-full ${color}`}>
      {pct}%
    </span>
  );
}

function MiniBar({ value, max = 1 }: { value: number | null; max?: number }) {
  if (value == null) return <div className="w-full h-2 bg-gray-100 rounded" />;
  const pct = Math.min(100, Math.round((value / max) * 100));
  const color = pct >= 80 ? "bg-green-500" : pct >= 60 ? "bg-yellow-400" : "bg-red-400";
  return (
    <div className="w-full h-2 bg-gray-100 rounded overflow-hidden">
      <div className={`h-full ${color} rounded`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function EvalDashboard() {
  const [stats, setStats] = useState<EvalStats | null>(null);
  const [trend, setTrend] = useState<TrendDay[]>([]);
  const [breakdown, setBreakdown] = useState<HashGroup[]>([]);
  const [recent, setRecent] = useState<EvalRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedRow, setSelectedRow] = useState<EvalRow | null>(null);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchMsg, setBatchMsg] = useState<string | null>(null);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    try {
      const [s, t, b, r] = await Promise.all([
        fetch(`${API_URL}/api/eval/stats`).then((x) => x.json()),
        fetch(`${API_URL}/api/eval/trend`).then((x) => x.json()),
        fetch(`${API_URL}/api/eval/breakdown`).then((x) => x.json()),
        fetch(`${API_URL}/api/eval/recent?limit=100`).then((x) => x.json()),
      ]);
      setStats(s);
      setTrend(Array.isArray(t) ? t : []);
      setBreakdown(Array.isArray(b) ? b : []);
      setRecent(Array.isArray(r) ? r : []);
    } catch {
      /* silencieux */
    } finally {
      setLoading(false);
    }
  }, []);

  // Polling queue status quand un batch est en cours
  useEffect(() => {
    if (!batchRunning) return;
    const poll = async () => {
      try {
        const res = await apiFetch(`${API_URL}/api/eval/queue-status`);
        const data = await res.json();
        if (!data.running) {
          setBatchRunning(false);
          setBatchMsg(null);
          fetchAll();
        }
      } catch { /* silencieux */ }
    };
    const id = setInterval(poll, 4000);
    return () => clearInterval(id);
  }, [batchRunning, fetchAll]);

  const handleRunBatch = async () => {
    setBatchMsg(null);
    try {
      const res = await apiFetch(`${API_URL}/api/eval/run-batch?limit=5`, { method: "POST" });
      const data = await res.json();
      if (data.started) {
        setBatchRunning(true);
        setBatchMsg(`Évaluation démarrée — ${data.pending_before} items en attente (traitement de ${data.limit} max)`);
      } else {
        setBatchMsg(data.reason ?? "Non démarré");
      }
    } catch {
      setBatchMsg("Erreur de connexion");
    }
  };

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">Évaluation RAG</h1>
            <p className="text-sm text-gray-500 mt-0.5">
              RAGAS on-demand · judge : qwen3.5:0.8b · les données s&apos;accumulent à chaque requête
            </p>
          </div>
          <div className="flex gap-3 items-center">
            <button
              onClick={handleRunBatch}
              disabled={batchRunning}
              className={`text-sm px-4 py-2 rounded-lg border font-medium transition-colors ${
                batchRunning
                  ? "bg-blue-50 border-blue-200 text-blue-500 cursor-not-allowed"
                  : stats?.pending
                    ? "bg-blue-600 border-blue-600 text-white hover:bg-blue-700"
                    : "bg-white border-gray-200 text-gray-400 cursor-not-allowed"
              }`}
            >
              {batchRunning
                ? "⏳ Évaluation en cours…"
                : `▶ Évaluer (${stats?.pending ?? 0} en attente)`}
            </button>
            <button
              onClick={fetchAll}
              className="text-sm px-4 py-2 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 text-gray-700"
            >
              ↻ Rafraîchir
            </button>
            <Link href="/admin" className="text-sm text-blue-600 hover:underline">
              ← Admin
            </Link>
          </div>
        </div>

        {batchMsg && (
          <div className={`mb-4 p-3 rounded-lg text-sm ${
            batchRunning ? "bg-blue-50 text-blue-700" : "bg-gray-50 text-gray-600"
          }`}>
            {batchMsg}
          </div>
        )}

        {loading && <p className="text-gray-500 text-sm mb-4">Chargement…</p>}

        {/* Stats cards */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {[
              { label: "Requêtes évaluées", value: String(stats.total ?? 0), sub: `${stats.errors ?? 0} erreurs · ${stats.pending ?? 0} en attente` },
              { label: "Fidélité (avg)", value: fmt(stats.avg_faithfulness), sub: "Faithfulness" },
              { label: "Pertinence (avg)", value: fmt(stats.avg_answer_relevancy), sub: "Answer Relevancy" },
              { label: "Précision ctx (avg)", value: fmt(stats.avg_context_precision), sub: "Context Precision" },
              { label: "Retrieval (avg)", value: `${fmt(stats.avg_retrieval_ms, 0)} ms`, sub: "Recherche vectorielle" },
              { label: "Éval (avg)", value: `${fmt(stats.avg_eval_ms, 0)} ms`, sub: "Temps RAGAS judge" },
            ].map((card) => (
              <div key={card.label} className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
                <p className="text-xs text-gray-500 mb-1">{card.label}</p>
                <p className="text-2xl font-bold text-gray-800">{card.value}</p>
                <p className="text-xs text-gray-400 mt-0.5">{card.sub}</p>
              </div>
            ))}
          </div>
        )}

        {/* Trend chart (sparkline table) */}
        {trend.length > 0 && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 mb-6">
            <h2 className="text-sm font-semibold text-gray-700 mb-4">Tendance journalière (30 jours)</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-400 border-b">
                    <th className="text-left pb-2 pr-4">Jour</th>
                    <th className="text-right pb-2 pr-4">Requêtes</th>
                    <th className="pb-2 pr-4 text-left w-32">Fidélité</th>
                    <th className="pb-2 pr-4 text-left w-32">Pertinence</th>
                    <th className="pb-2 text-left w-32">Précision ctx</th>
                  </tr>
                </thead>
                <tbody>
                  {trend.map((d) => (
                    <tr key={d.day} className="border-b border-gray-50 hover:bg-gray-50">
                      <td className="py-1.5 pr-4 font-mono text-gray-600">{d.day}</td>
                      <td className="py-1.5 pr-4 text-right text-gray-600">{d.count}</td>
                      <td className="py-1.5 pr-4">
                        <div className="flex items-center gap-2">
                          <MiniBar value={d.faithfulness} />
                          <span className="w-8 text-right text-gray-600">{fmt(d.faithfulness)}</span>
                        </div>
                      </td>
                      <td className="py-1.5 pr-4">
                        <div className="flex items-center gap-2">
                          <MiniBar value={d.answer_relevancy} />
                          <span className="w-8 text-right text-gray-600">{fmt(d.answer_relevancy)}</span>
                        </div>
                      </td>
                      <td className="py-1.5">
                        <div className="flex items-center gap-2">
                          <MiniBar value={d.context_precision} />
                          <span className="w-8 text-right text-gray-600">{fmt(d.context_precision)}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Hash breakdown */}
        {breakdown.length > 0 && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 mb-6">
            <h2 className="text-sm font-semibold text-gray-700 mb-4">Comparaison configurations</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-400 border-b">
                    <th className="text-left pb-2 pr-3">Pipeline</th>
                    <th className="text-left pb-2 pr-3">Search</th>
                    <th className="text-right pb-2 pr-3">N</th>
                    <th className="text-center pb-2 pr-3">Fidélité</th>
                    <th className="text-center pb-2 pr-3">Pertinence</th>
                    <th className="text-center pb-2 pr-3">Précision</th>
                    <th className="text-left pb-2">Dernière eval</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.map((g) => (
                    <tr key={`${g.pipeline_hash}-${g.search_hash}`} className="border-b border-gray-50 hover:bg-gray-50">
                      <td className="py-1.5 pr-3 font-mono text-gray-600">{g.pipeline_hash}</td>
                      <td className="py-1.5 pr-3 font-mono text-gray-600">{g.search_hash}</td>
                      <td className="py-1.5 pr-3 text-right text-gray-600">{g.count}</td>
                      <td className="py-1.5 pr-3 text-center"><ScoreBadge value={g.faithfulness} /></td>
                      <td className="py-1.5 pr-3 text-center"><ScoreBadge value={g.answer_relevancy} /></td>
                      <td className="py-1.5 pr-3 text-center"><ScoreBadge value={g.context_precision} /></td>
                      <td className="py-1.5 text-gray-400">{g.last_seen?.slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Recent evals table */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">
            Évaluations récentes ({recent.length})
          </h2>
          {recent.length === 0 ? (
            <p className="text-sm text-gray-400">
              Aucune évaluation. Pose une question dans le chat pour démarrer l&apos;accumulation de données.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-400 border-b">
                    <th className="text-left pb-2 pr-3">Horodatage</th>
                    <th className="text-left pb-2 pr-3">Collection</th>
                    <th className="text-left pb-2 pr-3 max-w-[200px]">Question</th>
                    <th className="text-center pb-2 pr-3">Fidélité</th>
                    <th className="text-center pb-2 pr-3">Pertinence</th>
                    <th className="text-center pb-2 pr-3">Précision</th>
                    <th className="text-right pb-2 pr-3">Retrieval</th>
                    <th className="text-right pb-2">Éval</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map((row) => (
                    <tr
                      key={row.id}
                      className="border-b border-gray-50 hover:bg-blue-50 cursor-pointer"
                      onClick={() => setSelectedRow(row)}
                    >
                      <td className="py-1.5 pr-3 text-gray-400 whitespace-nowrap">{row.timestamp}</td>
                      <td className="py-1.5 pr-3 text-gray-600">{row.collection}</td>
                      <td className="py-1.5 pr-3 text-gray-700 max-w-[200px] truncate" title={row.question}>
                        {row.question}
                      </td>
                      <td className="py-1.5 pr-3 text-center"><ScoreBadge value={row.faithfulness} /></td>
                      <td className="py-1.5 pr-3 text-center"><ScoreBadge value={row.answer_relevancy} /></td>
                      <td className="py-1.5 pr-3 text-center">
                        {row.eval_error
                          ? <span className="text-red-400 text-[10px]" title={row.eval_error}>Erreur</span>
                          : <ScoreBadge value={row.context_precision} />
                        }
                      </td>
                      <td className="py-1.5 pr-3 text-right text-gray-500">
                        {row.retrieval_ms != null ? `${Math.round(row.retrieval_ms)} ms` : "—"}
                      </td>
                      <td className="py-1.5 text-right text-gray-400">
                        {row.eval_ms != null ? `${Math.round(row.eval_ms)} ms` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Detail modal */}
        {selectedRow && (
          <div
            className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedRow(null)}
          >
            <div
              className="bg-white rounded-2xl shadow-xl max-w-2xl w-full p-6 max-h-[80vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex justify-between items-start mb-4">
                <h3 className="font-semibold text-gray-800">Détail évaluation #{selectedRow.id}</h3>
                <button onClick={() => setSelectedRow(null)} className="text-gray-400 hover:text-gray-700 text-xl leading-none">×</button>
              </div>

              <div className="space-y-3 text-sm">
                <div>
                  <span className="text-gray-500 text-xs uppercase tracking-wide">Question</span>
                  <p className="mt-0.5 text-gray-800">{selectedRow.question}</p>
                </div>
                <div>
                  <span className="text-gray-500 text-xs uppercase tracking-wide">Réponse (preview)</span>
                  <p className="mt-0.5 text-gray-700 text-xs bg-gray-50 p-2 rounded">{selectedRow.answer_preview}</p>
                </div>
                <div>
                  <span className="text-gray-500 text-xs uppercase tracking-wide">Contexte (preview)</span>
                  <p className="mt-0.5 text-gray-600 text-xs bg-gray-50 p-2 rounded font-mono whitespace-pre-wrap">{selectedRow.context_preview}</p>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    ["Fidélité", selectedRow.faithfulness],
                    ["Pertinence", selectedRow.answer_relevancy],
                    ["Précision ctx", selectedRow.context_precision],
                  ].map(([label, val]) => (
                    <div key={String(label)} className="text-center bg-gray-50 rounded-lg p-3">
                      <p className="text-gray-500 text-xs">{label}</p>
                      <p className="text-xl font-bold text-gray-800 mt-1">
                        {val != null ? (val as number).toFixed(3) : "—"}
                      </p>
                    </div>
                  ))}
                </div>
                <div className="flex gap-4 text-xs text-gray-500">
                  <span>Pipeline: <code className="font-mono">{selectedRow.pipeline_hash}</code></span>
                  <span>Search: <code className="font-mono">{selectedRow.search_hash}</code></span>
                  <span>Modèle juge: {selectedRow.eval_model}</span>
                </div>
                {selectedRow.eval_error && (
                  <div className="bg-red-50 text-red-700 text-xs p-2 rounded">
                    <strong>Erreur :</strong> {selectedRow.eval_error}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
