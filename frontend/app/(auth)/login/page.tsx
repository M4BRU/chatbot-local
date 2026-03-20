"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Step = "credentials" | "totp";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [totpCode, setTotpCode] = useState("");
  const [preAuthToken, setPreAuthToken] = useState("");
  const [step, setStep] = useState<Step>("credentials");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleCredentials(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const body = new URLSearchParams({ username: email, password });
      const res = await fetch(`${API_URL}/auth/jwt/login`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: body.toString(),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.detail || "Identifiants incorrects");
        return;
      }

      // Si TOTP activé : le backend retourne { totp_required: true, pre_auth_token }
      // Si pas de TOTP : le backend pose le cookie directement (corps vide)
      const data = await res.json().catch(() => ({}));
      if (data.totp_required) {
        setPreAuthToken(data.pre_auth_token);
        setStep("totp");
      } else {
        router.push("/");
        router.refresh();
      }
    } catch {
      setError("Erreur réseau — vérifiez votre connexion");
    } finally {
      setLoading(false);
    }
  }

  async function handleTotp(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/auth/totp/verify`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pre_auth_token: preAuthToken, code: totpCode }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.detail || "Code TOTP invalide");
        return;
      }

      router.push("/");
      router.refresh();
    } catch {
      setError("Erreur réseau — vérifiez votre connexion");
    } finally {
      setLoading(false);
    }
  }

  if (step === "totp") {
    return (
      <div className="w-full max-w-sm space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-semibold">Code d&apos;authentification</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Entrez le code généré par votre application 2FA
          </p>
        </div>

        <form onSubmit={handleTotp} className="space-y-4">
          <div className="space-y-2">
            <label htmlFor="totp" className="text-sm font-medium">
              Code à 6 chiffres
            </label>
            <input
              id="totp"
              type="text"
              inputMode="numeric"
              pattern="[0-9]{6}"
              maxLength={6}
              required
              autoComplete="one-time-code"
              autoFocus
              value={totpCode}
              onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, ""))}
              className="w-full px-3 py-2 border border-input rounded-md bg-background text-sm text-center tracking-[0.5em] focus:outline-none focus:ring-2 focus:ring-ring"
              placeholder="000000"
            />
          </div>

          {error && <p className="text-sm text-destructive">{error}</p>}

          <button
            type="submit"
            disabled={loading || totpCode.length !== 6}
            className="w-full py-2 px-4 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Vérification…" : "Valider"}
          </button>

          <button
            type="button"
            onClick={() => {
              setStep("credentials");
              setError("");
              setTotpCode("");
            }}
            className="w-full text-sm text-muted-foreground hover:text-foreground"
          >
            ← Retour
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="w-full max-w-sm space-y-6">
      <div className="text-center">
        <h1 className="text-2xl font-semibold">Connexion</h1>
        <p className="text-sm text-muted-foreground mt-1">Chatbot VLM</p>
      </div>

      <form onSubmit={handleCredentials} className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="email" className="text-sm font-medium">
            Email
          </label>
          <input
            id="email"
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full px-3 py-2 border border-input rounded-md bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            placeholder="vous@example.com"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="password" className="text-sm font-medium">
            Mot de passe
          </label>
          <input
            id="password"
            type="password"
            required
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full px-3 py-2 border border-input rounded-md bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>

        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full py-2 px-4 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Connexion…" : "Se connecter"}
        </button>
      </form>
    </div>
  );
}
