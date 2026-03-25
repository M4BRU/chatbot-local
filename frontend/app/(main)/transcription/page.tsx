"use client";

import { useCallback, useRef, useState } from "react";
import {
  CheckCircle2,
  Copy,
  FileAudio,
  Mic,
  RotateCcw,
  Upload,
  XCircle,
} from "lucide-react";
import { streamTranscription } from "@/app/lib/api";
import type {
  TranscriptionSegment,
  TranscriptionSSEEvent,
  TranscriptionStatus,
  TranscriptionSummary,
} from "@/app/lib/types";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";

const MAX_SIZE_MB = 300;
const ALLOWED_TYPES = [".mp3", ".wav", ".m4a"];

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m} min ${s} s` : `${s} s`;
}

// ── Step indicator ──────────────────────────────────────────────────────────

const STEPS = [
  { key: "transcribing", label: "Transcription" },
  { key: "summarizing", label: "Résumé" },
  { key: "done", label: "Terminé" },
] as const;

function ProgressSteps({ status }: { status: TranscriptionStatus | null }) {
  if (!status) return null;
  const idx = STEPS.findIndex((s) => s.key === status);

  return (
    <div className="flex items-center gap-3 py-4">
      {STEPS.map((step, i) => {
        const isActive =
          step.key === status ||
          (status === "transcribed" && step.key === "transcribing");
        const isDone =
          (status === "done" && true) ||
          (idx > i) ||
          (status === "transcribed" && step.key === "transcribing");
        const isError = status === "error" && i === idx;

        return (
          <div key={step.key} className="flex items-center gap-2">
            {i > 0 && (
              <div
                className={cn(
                  "w-8 h-px",
                  isDone ? "bg-emerald-500" : "bg-border"
                )}
              />
            )}
            <div className="flex items-center gap-1.5">
              {isDone ? (
                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
              ) : isError ? (
                <XCircle className="h-4 w-4 text-destructive" />
              ) : isActive ? (
                <Spinner className="h-4 w-4 text-primary" />
              ) : (
                <div className="h-4 w-4 rounded-full border border-muted-foreground/30" />
              )}
              <span
                className={cn(
                  "text-sm",
                  isDone
                    ? "text-emerald-600 font-medium"
                    : isActive
                      ? "text-foreground font-medium"
                      : "text-muted-foreground"
                )}
              >
                {step.label}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Summary display ─────────────────────────────────────────────────────────

function SummarySection({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border bg-card p-4">
      <h3 className="flex items-center gap-2 font-semibold text-sm mb-3">
        {icon}
        {title}
      </h3>
      {children}
    </div>
  );
}

function SummaryDisplay({ summary }: { summary: TranscriptionSummary }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {/* Décisions */}
      <SummarySection title="Décisions" icon={<CheckCircle2 className="h-4 w-4 text-emerald-500" />}>
        {summary.decisions.length > 0 ? (
          <ul className="space-y-1 text-sm">
            {summary.decisions.map((d, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-muted-foreground">•</span>
                <span>{d}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted-foreground italic">Aucune décision détectée</p>
        )}
      </SummarySection>

      {/* Actions */}
      <SummarySection title="Actions" icon={<RotateCcw className="h-4 w-4 text-blue-500" />}>
        {summary.actions.length > 0 ? (
          <div className="space-y-2">
            {summary.actions.map((a, i) => (
              <div key={i} className="text-sm border-l-2 border-blue-200 pl-3">
                <p>{a.description}</p>
                <div className="flex gap-3 text-xs text-muted-foreground mt-0.5">
                  {a.responsable && <span>👤 {a.responsable}</span>}
                  {a.deadline && <span>📅 {a.deadline}</span>}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground italic">Aucune action détectée</p>
        )}
      </SummarySection>

      {/* Participants */}
      <SummarySection title="Participants" icon={<Mic className="h-4 w-4 text-violet-500" />}>
        {summary.participants.length > 0 ? (
          <div className="flex flex-wrap gap-1.5">
            {summary.participants.map((p, i) => (
              <span
                key={i}
                className="inline-flex items-center rounded-full bg-violet-100 px-2.5 py-0.5 text-xs font-medium text-violet-800 dark:bg-violet-900/30 dark:text-violet-300"
              >
                {p}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground italic">Aucun participant détecté</p>
        )}
      </SummarySection>

      {/* Points clés */}
      <SummarySection title="Points clés" icon={<FileAudio className="h-4 w-4 text-amber-500" />}>
        {summary.points_cles.length > 0 ? (
          <ul className="space-y-1 text-sm">
            {summary.points_cles.map((p, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-muted-foreground">•</span>
                <span>{p}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted-foreground italic">Aucun point clé détecté</p>
        )}
      </SummarySection>
    </div>
  );
}

// ── Main page ───────────────────────────────────────────────────────────────

export default function TranscriptionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<TranscriptionStatus | null>(null);
  const [step, setStep] = useState("");
  const [transcript, setTranscript] = useState("");
  const [segments, setSegments] = useState<TranscriptionSegment[]>([]);
  const [duration, setDuration] = useState(0);
  const [summary, setSummary] = useState<TranscriptionSummary | null>(null);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const isProcessing = status === "transcribing" || status === "summarizing" || status === "transcribed";

  const reset = useCallback(() => {
    setFile(null);
    setStatus(null);
    setStep("");
    setTranscript("");
    setSegments([]);
    setDuration(0);
    setSummary(null);
    setError("");
    setCopied(false);
  }, []);

  const handleFile = useCallback((f: File | null) => {
    if (!f) return;
    const ext = f.name.substring(f.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED_TYPES.includes(ext)) {
      setError(`Format non supporté. Formats acceptés : ${ALLOWED_TYPES.join(", ")}`);
      return;
    }
    if (f.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`Fichier trop volumineux (${(f.size / 1024 / 1024).toFixed(0)} MB). Maximum : ${MAX_SIZE_MB} MB.`);
      return;
    }
    setError("");
    setFile(f);
  }, []);

  const handleTranscribe = useCallback(async () => {
    if (!file || isProcessing) return;

    setError("");
    setTranscript("");
    setSegments([]);
    setDuration(0);
    setSummary(null);
    setStatus("transcribing");
    setStep("Transcription en cours…");

    try {
      for await (const event of streamTranscription(file)) {
        if (event.error) {
          setStatus("error");
          setError(event.error);
          break;
        }
        if (event.status) setStatus(event.status);
        if (event.step) setStep(event.step);
        if (event.transcript) setTranscript(event.transcript);
        if (event.segments) setSegments(event.segments);
        if (event.duration_seconds !== undefined) setDuration(event.duration_seconds);
        if (event.summary) setSummary(event.summary);
      }
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [file, isProcessing]);

  const copyTranscript = useCallback(() => {
    navigator.clipboard.writeText(transcript);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [transcript]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b bg-background shrink-0">
        <SidebarTrigger />
        <Mic className="h-5 w-5 text-violet-500" />
        <span className="font-semibold">Transcription</span>
        {status === "done" && (
          <span className="ml-auto text-xs text-muted-foreground">
            {formatDuration(duration)} — {segments.length} segments
          </span>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-[780px] w-full mx-auto px-4 py-6 space-y-6">

          {/* ── File drop zone ────────────────────────────────────────── */}
          {!isProcessing && status !== "done" && (
            <div
              className={cn(
                "relative rounded-xl border-2 border-dashed p-8 text-center transition-colors cursor-pointer",
                dragOver
                  ? "border-primary bg-primary/5"
                  : file
                    ? "border-emerald-300 bg-emerald-50/50 dark:bg-emerald-900/10"
                    : "border-muted-foreground/25 hover:border-muted-foreground/50",
              )}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleFile(e.dataTransfer.files[0] ?? null);
              }}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp3,.wav,.m4a"
                className="hidden"
                onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
              />

              {file ? (
                <div className="space-y-2">
                  <FileAudio className="h-10 w-10 mx-auto text-emerald-500" />
                  <p className="font-medium">{file.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(file.size / 1024 / 1024).toFixed(1)} MB
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  <Upload className="h-10 w-10 mx-auto text-muted-foreground/50" />
                  <p className="text-muted-foreground">
                    Glissez un fichier audio ici ou cliquez pour parcourir
                  </p>
                  <p className="text-xs text-muted-foreground">
                    MP3, WAV, M4A — max {MAX_SIZE_MB} MB
                  </p>
                </div>
              )}
            </div>
          )}

          {/* ── Transcribe button ─────────────────────────────────────── */}
          {file && !isProcessing && status !== "done" && (
            <button
              onClick={handleTranscribe}
              className="w-full py-3 rounded-lg bg-violet-600 text-white font-medium hover:bg-violet-700 transition-colors"
            >
              Transcrire
            </button>
          )}

          {/* ── Error ─────────────────────────────────────────────────── */}
          {error && (
            <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
              <p className="text-sm text-destructive">{error}</p>
              <button
                onClick={reset}
                className="mt-2 text-sm text-destructive underline hover:no-underline"
              >
                Réessayer
              </button>
            </div>
          )}

          {/* ── Progress steps ────────────────────────────────────────── */}
          {status && status !== "error" && <ProgressSteps status={status} />}

          {/* ── Processing indicator ──────────────────────────────────── */}
          {isProcessing && (
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              <Spinner className="h-4 w-4" />
              <span>{step}</span>
            </div>
          )}

          {/* ── Transcript ────────────────────────────────────────────── */}
          {transcript && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-sm">Transcription brute</h2>
                <div className="flex items-center gap-2">
                  {duration > 0 && (
                    <span className="text-xs bg-muted px-2 py-0.5 rounded">
                      {formatDuration(duration)}
                    </span>
                  )}
                  <button
                    onClick={copyTranscript}
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <Copy className="h-3 w-3" />
                    {copied ? "Copié !" : "Copier"}
                  </button>
                </div>
              </div>
              <div className="rounded-lg border bg-muted/50 p-4 max-h-64 overflow-y-auto">
                <p className="text-sm whitespace-pre-wrap leading-relaxed">{transcript}</p>
              </div>
            </div>
          )}

          {/* ── Summary ───────────────────────────────────────────────── */}
          {summary && (
            <div className="space-y-3">
              <h2 className="font-semibold text-sm">Résumé structuré</h2>
              <SummaryDisplay summary={summary} />
            </div>
          )}

          {/* ── New file button ───────────────────────────────────────── */}
          {status === "done" && (
            <button
              onClick={reset}
              className="w-full py-2.5 rounded-lg border border-border text-sm font-medium hover:bg-muted transition-colors"
            >
              Nouveau fichier
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
