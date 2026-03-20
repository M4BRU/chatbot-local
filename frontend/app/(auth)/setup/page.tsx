"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface TotpSetup {
  secret: string;
  otpauth_uri: string;
  qr_data_url: string;
}

export default function SetupPage() {
  const router = useRouter();
  const [setupData, setSetupData] = useState<TotpSetup | null>(null);
  const [code, setCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function startSetup() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_URL}/auth/totp/setup`, { credentials: "include" });
      if (!res.ok) throw new Error();
      setSetupData(await res.json());
    } catch {
      setError("Erreur lors de la génération du QR code");
    } finally {
      setLoading(false);
    }
  }

  async function handleEnable(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_URL}/auth/totp/enable`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        setError(d.detail || "Code invalide");
        return;
      }
      // 2FA activé → accès débloqué, redirection directe vers le chat
      router.push("/");
      router.refresh();
    } catch {
      setError("Erreur réseau");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="w-full max-w-md space-y-6">
      <div className="text-center space-y-1">
        <h1 className="text-2xl font-semibold">Sécurisez votre compte</h1>
        <p className="text-sm text-muted-foreground">
          Avant d&apos;accéder à l&apos;application, configurez
          l&apos;authentification à deux facteurs.
        </p>
      </div>

      {!setupData ? (
        <div className="space-y-4">
          <div className="rounded-md border px-4 py-3 text-sm text-muted-foreground space-y-2">
            <p>
              <strong>Ce dont vous avez besoin :</strong> une application
              d&apos;authentification sur votre téléphone.
            </p>
            <p>
              Nous recommandons <strong>2FAS</strong> (iOS / Android, gratuit) ou
              tout autre application compatible TOTP — Google Authenticator,
              Microsoft Authenticator, etc.
            </p>
          </div>
          <button
            onClick={startSetup}
            disabled={loading}
            className="w-full py-2 px-4 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
          >
            {loading ? "Chargement…" : "Générer mon QR code"}
          </button>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
      ) : (
        <div className="space-y-5">
          <p className="text-sm text-muted-foreground">
            Scannez ce QR code avec votre application, puis entrez le code à 6
            chiffres pour confirmer.
          </p>

          <div className="flex justify-center">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={setupData.qr_data_url}
              alt="QR Code TOTP"
              width={180}
              height={180}
              className="border rounded-lg bg-white p-2"
            />
          </div>

          <details className="text-xs text-muted-foreground">
            <summary className="cursor-pointer select-none">
              Entrer le code manuellement
            </summary>
            <p className="mt-1 font-mono break-all bg-muted px-2 py-1 rounded">
              {setupData.secret}
            </p>
          </details>

          <form onSubmit={handleEnable} className="space-y-3">
            <input
              type="text"
              inputMode="numeric"
              pattern="[0-9]{6}"
              maxLength={6}
              required
              autoFocus
              placeholder="Code à 6 chiffres"
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
              className="w-full px-3 py-2 border border-input rounded-md bg-background text-sm text-center tracking-[0.5em] focus:outline-none focus:ring-2 focus:ring-ring"
            />
            {error && <p className="text-sm text-destructive">{error}</p>}
            <button
              type="submit"
              disabled={loading || code.length !== 6}
              className="w-full py-2 px-4 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Activation…" : "Activer et accéder à l'application"}
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
