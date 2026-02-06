"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { streamChat, fetchCollections } from "../lib/api";
import type { ChatMessage, ChatSource } from "../lib/types";

function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
  collection: string;
  collections: string[];
  onCollectionChange: (collection: string) => void;
}

function ChatInput({ onSend, disabled, collection, collections, onCollectionChange }: ChatInputProps) {
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled && collection) {
      onSend(input.trim());
      setInput("");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 p-4 border-t border-gray-200">
      <select
        value={collection}
        onChange={(e) => onCollectionChange(e.target.value)}
        className="px-3 py-2 border border-gray-300 rounded-lg w-48 text-sm bg-white"
      >
        <option value="">-- Collection --</option>
        {collections.map((c) => (
          <option key={c} value={c}>{c}</option>
        ))}
      </select>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Posez votre question..."
        disabled={disabled || !collection}
        className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
      />
      <button
        type="submit"
        disabled={disabled || !input.trim() || !collection}
        className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
      >
        {disabled ? "..." : "Envoyer"}
      </button>
    </form>
  );
}

function Sources({ sources }: { sources: ChatSource[] }) {
  if (!sources.length) return null;

  return (
    <div className="mt-3 pt-3 border-t border-gray-200">
      <p className="text-xs font-semibold text-gray-500 mb-1">Sources:</p>
      <ul className="text-xs text-gray-600 space-y-0.5">
        {sources.map((s, i) => (
          <li key={i}>
            {s.fichier} - p.{s.page} (score: {s.score})
          </li>
        ))}
      </ul>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[80%] px-4 py-3 rounded-2xl ${
          isUser
            ? "bg-blue-600 text-white rounded-br-md"
            : "bg-gray-100 text-gray-800 rounded-bl-md"
        }`}
      >
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content || (message.isStreaming ? "..." : "")}
            </ReactMarkdown>
          </div>
        )}
        {message.sources && <Sources sources={message.sources} />}
      </div>
    </div>
  );
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [collection, setCollection] = useState("");
  const [collections, setCollections] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch collections on mount
  useEffect(() => {
    fetchCollections().then((cols) => {
      setCollections(cols);
      if (cols.length > 0 && !collection) {
        setCollection(cols[0]);
      }
    });
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (content: string) => {
    setError(null);
    const userMessage: ChatMessage = {
      id: generateId(),
      role: "user",
      content,
    };

    const assistantMessage: ChatMessage = {
      id: generateId(),
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    // Prepare history from existing messages (exclude sources for smaller payload)
    const history = messages
      .filter((m) => m.content && !m.isStreaming)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    try {
      for await (const event of streamChat(content, collection, "defaut", history)) {
        if (event.error) {
          setError(event.error);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessage.id
                ? { ...m, content: `Erreur: ${event.error}`, isStreaming: false }
                : m
            )
          );
          break;
        }

        if (event.token) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessage.id ? { ...m, content: m.content + event.token } : m
            )
          );
        }

        if (event.done && event.sources) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessage.id
                ? { ...m, sources: event.sources, isStreaming: false }
                : m
            )
          );
        }
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Erreur de connexion";
      setError(errorMsg);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessage.id
            ? { ...m, content: `Erreur: ${errorMsg}`, isStreaming: false }
            : m
        )
      );
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto">
      <header className="p-4 border-b border-gray-200 flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold text-gray-800">chatbot-local</h1>
          <p className="text-sm text-gray-500">RAG chatbot pour VLM Robotics</p>
        </div>
        <Link href="/admin" className="text-blue-600 hover:underline text-sm">
          Admin
        </Link>
      </header>

      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-lg">Bienvenue!</p>
            <p className="text-sm mt-2">
              {collections.length === 0
                ? "Aucune collection disponible. Allez dans Admin pour en créer une."
                : "Sélectionnez une collection et posez votre question"}
            </p>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {error && (
          <div className="text-center text-red-500 text-sm py-2">{error}</div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <ChatInput
        onSend={handleSend}
        disabled={isStreaming}
        collection={collection}
        collections={collections}
        onCollectionChange={setCollection}
      />
    </div>
  );
}
