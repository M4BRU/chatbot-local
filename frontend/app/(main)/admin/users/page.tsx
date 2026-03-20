"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const apiFetch = (input: RequestInfo | URL, init?: RequestInit) =>
  fetch(input, { credentials: "include", ...init });

const ROLES = ["STANDARD", "COMMERCIAL", "ADMIN", "DEV"];

interface User {
  id: string;
  email: string;
  role: string;
  is_active: boolean;
  totp_enabled: boolean;
  created_at?: string;
}

export default function UsersPage() {
  const [users, setUsers] = useState<User[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("STANDARD");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  async function fetchUsers() {
    try {
      const res = await apiFetch(`${API_URL}/api/v1/users`);
      if (res.ok) setUsers(await res.json());
    } finally {
      setLoadingList(false);
    }
  }

  useEffect(() => { fetchUsers(); }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setCreating(true);
    setError("");
    setSuccess("");
    try {
      const res = await apiFetch(`${API_URL}/api/v1/users`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, role }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        setError(d.detail || "Erreur lors de la création");
        return;
      }
      setSuccess(`Compte créé pour ${email}`);
      setEmail("");
      setPassword("");
      setRole("STANDARD");
      fetchUsers();
    } catch {
      setError("Erreur réseau");
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto py-8 px-4 space-y-8">
      <h1 className="text-2xl font-semibold">Gestion des utilisateurs</h1>

      {/* Formulaire création */}
      <div className="space-y-4 border rounded-lg p-6">
        <h2 className="text-base font-medium">Créer un compte</h2>

        {success && <p className="text-sm text-green-600 dark:text-green-400">{success}</p>}
        {error && <p className="text-sm text-destructive">{error}</p>}

        <form onSubmit={handleCreate} className="space-y-3">
          <div className="flex gap-3">
            <input
              type="email"
              required
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="flex-1 px-3 py-2 border border-input rounded-md bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <input
              type="password"
              required
              placeholder="Mot de passe"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="flex-1 px-3 py-2 border border-input rounded-md bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <select
              value={role}
              onChange={(e) => setRole(e.target.value)}
              className="px-3 py-2 border border-input rounded-md bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            >
              {ROLES.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </div>
          <button
            type="submit"
            disabled={creating}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
          >
            {creating ? "Création…" : "Créer le compte"}
          </button>
        </form>
      </div>

      {/* Liste utilisateurs */}
      <div className="space-y-3">
        <h2 className="text-base font-medium">Utilisateurs ({users.length})</h2>
        {loadingList ? (
          <p className="text-sm text-muted-foreground">Chargement…</p>
        ) : (
          <div className="border rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left px-4 py-2 font-medium">Email</th>
                  <th className="text-left px-4 py-2 font-medium">Rôle</th>
                  <th className="text-left px-4 py-2 font-medium">2FA</th>
                  <th className="text-left px-4 py-2 font-medium">Statut</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id} className="border-t">
                    <td className="px-4 py-2">{u.email}</td>
                    <td className="px-4 py-2">
                      <span className="px-2 py-0.5 rounded-full text-xs bg-muted font-mono">
                        {u.role}
                      </span>
                    </td>
                    <td className="px-4 py-2">
                      {u.totp_enabled
                        ? <span className="text-green-600 text-xs">✓ activé</span>
                        : <span className="text-muted-foreground text-xs">non configuré</span>}
                    </td>
                    <td className="px-4 py-2">
                      <span className={u.is_active ? "text-green-600 text-xs" : "text-destructive text-xs"}>
                        {u.is_active ? "actif" : "inactif"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
