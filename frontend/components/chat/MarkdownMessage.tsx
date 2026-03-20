"use client";

/**
 * Composants de rendu de messages partagés entre toutes les pages chat.
 *
 * Exports:
 *   UserBubble        — bulle message utilisateur
 *   AssistantMessage  — avatar + prose Markdown + sources + curseur streaming
 *                       accepte des children (badges, tool-call indicators, etc.)
 *   LoadingDots       — indicateur de chargement animé
 */

import { useState } from "react";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { Check, Copy } from "lucide-react";
import type { ChatSource, RagMetrics } from "@/app/lib/types";

// ─── Code block avec bouton copier ────────────────────────────────────────────

function CodeBlock({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLElement>) {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || "");
  const language = match?.[1] ?? "";
  const code = String(children).replace(/\n$/, "");

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Inline code
  if (!match) {
    return (
      <code
        className="bg-muted text-orange-400 dark:text-orange-300 px-1.5 py-0.5 rounded text-[0.85em] font-mono"
        {...props}
      >
        {children}
      </code>
    );
  }

  // Block code
  return (
    <div className="relative my-4 rounded-lg overflow-hidden border border-border">
      <div className="flex items-center justify-between px-4 py-2 bg-muted/60 border-b border-border">
        <span className="text-xs text-muted-foreground font-mono">{language}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? "Copié !" : "Copier"}
        </button>
      </div>
      <pre className="overflow-x-auto p-4 text-sm m-0 bg-[#0d1117]">
        <code className={className} {...props}>
          {children}
        </code>
      </pre>
    </div>
  );
}

// ─── Contenu Markdown rendu ────────────────────────────────────────────────────

function MarkdownContent({
  content,
  isStreaming,
  noCode,
}: {
  content: string;
  isStreaming?: boolean;
  /** Si true, les blocs de code sont rendus en texte brut (sans highlight ni cadre) */
  noCode?: boolean;
}) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const codeRenderer: any = noCode
    ? ({ children }: React.HTMLAttributes<HTMLElement>) => <span className="font-mono text-sm">{children}</span>
    : CodeBlock;

  return (
    <div className="prose prose-base dark:prose-invert max-w-none text-foreground leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={noCode ? [] : [rehypeHighlight]}
        components={{
          code: codeRenderer,
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:underline"
            >
              {children}
            </a>
          ),
        }}
      >
        {content || (isStreaming ? "\u200b" : "")}
      </ReactMarkdown>
      {isStreaming && (
        <span className="inline-block w-[2px] h-4 bg-foreground/60 animate-pulse align-middle ml-0.5" />
      )}
    </div>
  );
}

// ─── Bulle utilisateur ────────────────────────────────────────────────────────

export function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end mb-6">
      <div className="max-w-[80%] bg-muted text-foreground rounded-[18px] px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap">
        {content}
      </div>
    </div>
  );
}

// ─── Message assistant ────────────────────────────────────────────────────────

export function AssistantMessage({
  content,
  sources,
  metrics,
  isStreaming,
  noCode,
  children,
}: {
  content: string;
  sources?: ChatSource[];
  metrics?: RagMetrics;
  isStreaming?: boolean;
  /** Si true, les blocs de code sont rendus en texte brut (mode devis) */
  noCode?: boolean;
  /** Badges ou indicateurs affichés au-dessus du contenu (tool calls, step badges…) */
  children?: React.ReactNode;
}) {
  return (
    <div className="flex gap-3 mb-6">
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center mt-0.5 border border-border">
        <Image src="/logoVLM.png" alt="Assistant" width={18} height={18} className="object-contain" unoptimized />
      </div>

      {/* Contenu */}
      <div className="flex-1 min-w-0">
        {children}
        <MarkdownContent content={content} isStreaming={isStreaming} noCode={noCode} />

        {/* Sources */}
        {sources && sources.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {sources.map((src, i) => (
              <span
                key={i}
                className="text-xs text-muted-foreground bg-muted border border-border rounded px-2 py-1"
              >
                {src.fichier} · p.{src.page}
              </span>
            ))}
          </div>
        )}

        {/* Level-A RAG metrics */}
        {metrics && !isStreaming && (
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1">
            <span className="text-[11px] text-muted-foreground/60">
              {metrics.chunks_used} chunks
            </span>
            {metrics.top_score != null && (
              <span className="text-[11px] text-muted-foreground/60">
                score {metrics.top_score}–{metrics.min_score}
              </span>
            )}
            <span className="text-[11px] text-muted-foreground/60">
              {metrics.retrieval_ms} ms
            </span>
            <span className="text-[11px] text-muted-foreground/40 font-mono">
              {metrics.pipeline_hash}/{metrics.search_hash}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Loading dots ─────────────────────────────────────────────────────────────

export function LoadingDots() {
  return (
    <div className="flex gap-3 mb-6">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center border border-border">
        <Image src="/logoVLM.png" alt="Assistant" width={18} height={18} className="object-contain" unoptimized />
      </div>
      <div className="flex items-center gap-1.5 mt-2">
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:-0.3s]" />
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:-0.15s]" />
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce" />
      </div>
    </div>
  );
}
