"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useCurrentUser } from "@/app/lib/auth";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface TotpSetup {
  secret: string;
  otpauth_uri: string;
  qr_data_url: string;
}

export default function AccountPage() {
  const router = useRouter();
  const { user, loading: userLoading } = useCurrentUser();
  const [setupData, setSetupData] = useState<TotpSetup | null>(null);
  const [totpEnabled, setTotpEnabled] = useState<boolean>(false);
  const [enableCode, setEnableCode] = useState("");
  const [disableCode, setDisableCode] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Initialise l'état TOTP depuis le profil utilisateur
  useEffect(() => {
    if (user) {
      setTotpEnabled(user.totp_enabled ?? false);
    }
  }, [user]);

  async function startSetup() {
    setLoading(true);
    setError("");
    setMessage("");
    try {
      const res = await fetch(`${API_URL}/auth/totp/setup`, {
        credentials: "include",
      });
      if (!res.ok) throw new Error("Erreur setup");
      const data = await res.json();
      setSetupData(data);
    } catch {
      setError("Erreur lors de la configuration TOTP");
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
        body: JSON.stringify({ code: enableCode }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        setError(d.detail || "Code invalide");
        return;
      }
      setTotpEnabled(true);
      setSetupData(null);
      setEnableCode("");
      // Redirige vers le chat maintenant que le 2FA est actif
      router.push("/");
      router.refresh();
    } catch {
      setError("Erreur réseau");
    } finally {
      setLoading(false);
    }
  }

  async function handleDisable(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_URL}/auth/totp/disable`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: disableCode }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        setError(d.detail || "Code invalide");
        return;
      }
      setMessage("Authentification à deux facteurs désactivée.");
      setTotpEnabled(false);
      setDisableCode("");
    } catch {
      setError("Erreur réseau");
    } finally {
      setLoading(false);
    }
  }

  if (userLoading) {
    return <div className="max-w-lg mx-auto py-8 px-4 text-sm text-muted-foreground">Chargement…</div>;
  }

  return (
    <div className="max-w-lg mx-auto py-8 px-4 space-y-8">
      <h1 className="text-2xl font-semibold">Mon compte</h1>

      {/* Infos utilisateur */}
      <div className="space-y-1">
        <p className="text-sm text-muted-foreground">Email</p>
        <p className="text-sm font-medium">{user?.email}</p>
      </div>

      {/* Section 2FA */}
      <div className="space-y-4 border-t pt-6">
        <h2 className="text-lg font-medium">Authentification à deux facteurs (2FA)</h2>

        {message && (
          <p className="text-sm text-green-600 dark:text-green-400">{message}</p>
        )}
        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}

        {/* État : TOTP non activé, pas de setup en cours */}
        {!totpEnabled && !setupData && (
          <div className="space-y-4">
            <div className="rounded-md border border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950 px-4 py-3">
              <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
                Étape obligatoire avant d&apos;accéder à l&apos;application
              </p>
              <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                Pour des raisons de sécurité, vous devez configurer l&apos;authentification
                à deux facteurs. Installez l&apos;application <strong>2FAS</strong> (iOS / Android,
                gratuit) ou tout autre application compatible TOTP (Google Authenticator,
                Microsoft Authenticator…), puis cliquez sur le bouton ci-dessous.
              </p>
            </div>
            <button
              onClick={startSetup}
              disabled={loading}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
            >
              {loading ? "Chargement…" : "Configurer le 2FA"}
            </button>
          </div>
        )}

        {/* Étape setup : affichage QR + saisie code */}
        {setupData && (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Scannez ce QR code avec votre application 2FA, puis entrez le code généré pour activer.
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
                value={enableCode}
                onChange={(e) => setEnableCode(e.target.value.replace(/\D/g, ""))}
                className="w-full px-3 py-2 border border-input rounded-md bg-background text-sm text-center tracking-[0.5em] focus:outline-none focus:ring-2 focus:ring-ring"
              />
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={loading || enableCode.length !== 6}
                  className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
                >
                  {loading ? "Activation…" : "Activer"}
                </button>
                <button
                  type="button"
                  onClick={() => { setSetupData(null); setEnableCode(""); setError(""); }}
                  className="px-4 py-2 border rounded-md text-sm hover:bg-accent"
                >
                  Annuler
                </button>
              </div>
            </form>
          </div>
        )}

        {/* État : TOTP activé */}
        {totpEnabled && !setupData && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-green-500" />
              <span className="text-sm font-medium">2FA activé</span>
            </div>
            <form onSubmit={handleDisable} className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Entrez un code 2FA pour désactiver la protection.
              </p>
              <input
                type="text"
                inputMode="numeric"
                pattern="[0-9]{6}"
                maxLength={6}
                required
                placeholder="Code à 6 chiffres"
                value={disableCode}
                onChange={(e) => setDisableCode(e.target.value.replace(/\D/g, ""))}
                className="w-full px-3 py-2 border border-input rounded-md bg-background text-sm text-center tracking-[0.5em] focus:outline-none focus:ring-2 focus:ring-ring"
              />
              <button
                type="submit"
                disabled={loading || disableCode.length !== 6}
                className="px-4 py-2 bg-destructive text-destructive-foreground rounded-md text-sm font-medium hover:bg-destructive/90 disabled:opacity-50"
              >
                {loading ? "Désactivation…" : "Désactiver le 2FA"}
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}
