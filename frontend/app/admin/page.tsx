"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Document {
  nom: string;
  date: string;
  nb_chunks: number;
  nb_pages: number;
}

export default function AdminPage() {
  const [collections, setCollections] = useState<string[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

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

  useEffect(() => {
    fetchCollections();
  }, []);

  useEffect(() => {
    if (selectedCollection) {
      fetchDocuments(selectedCollection);
    }
  }, [selectedCollection]);

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

  // Upload document
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!selectedCollection || !e.target.files?.length) return;

    const file = e.target.files[0];
    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    setMessage(null);

    try {
      const res = await fetch(
        `${API_URL}/api/collections/${selectedCollection}/documents`,
        { method: "POST", body: formData }
      );

      const data = await res.json();
      if (res.ok) {
        setMessage({ type: "success", text: data.message });
        fetchDocuments(selectedCollection);
      } else {
        setMessage({ type: "error", text: data.detail || "Erreur upload" });
      }
    } catch {
      setMessage({ type: "error", text: "Erreur de connexion" });
    } finally {
      setUploading(false);
      e.target.value = "";
    }
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
          <Link href="/" className="text-blue-600 hover:underline">
            ← Retour au chat
          </Link>
        </div>

        {message && (
          <div
            className={`mb-4 p-3 rounded ${
              message.type === "success" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
            }`}
          >
            {message.text}
          </div>
        )}

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
                    <span>{name}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteCollection(name);
                      }}
                      className="text-red-500 hover:text-red-700 text-sm"
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
                  <span className="sr-only">Choisir un fichier</span>
                  <input
                    type="file"
                    accept=".pdf,.txt,.md,.docx"
                    onChange={handleUpload}
                    disabled={uploading}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
                  />
                </label>

                {uploading && (
                  <div className="mb-4 text-blue-600">Indexation en cours...</div>
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
      </div>
    </div>
  );
}
