import { NextRequest, NextResponse } from "next/server";

// BACKEND_INTERNAL_URL : URL interne Docker (http://backend:8000) pour les appels server-side
// NEXT_PUBLIC_API_URL : URL publique (https://localhost) pour le browser
const API_URL = process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Chemins publics (accessibles sans authentification)
const PUBLIC_PATHS = ["/login"];

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Laisser passer les chemins publics et les assets Next.js
  if (
    PUBLIC_PATHS.some((p) => pathname.startsWith(p)) ||
    pathname.startsWith("/_next") ||
    pathname.startsWith("/favicon")
  ) {
    return NextResponse.next();
  }

  // Vérifier le cookie d'authentification via le backend
  const cookie = request.cookies.get("access_token");
  if (!cookie?.value) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  try {
    const res = await fetch(`${API_URL}/auth/users/me`, {
      headers: { Cookie: `access_token=${cookie.value}` },
      // Pas de credentials: 'include' ici — on passe le cookie manuellement (server-side)
    });

    if (!res.ok) {
      return NextResponse.redirect(new URL("/login", request.url));
    }

    const user = await res.json();

    // 2FA obligatoire : forcer /setup (onboarding sans sidebar) si TOTP pas encore activé
    if (!user.totp_enabled && !pathname.startsWith("/setup")) {
      return NextResponse.redirect(new URL("/setup", request.url));
    }

    // Protéger /admin : ADMIN et DEV uniquement
    if (pathname.startsWith("/admin") && !["ADMIN", "DEV"].includes(user?.role)) {
      return NextResponse.redirect(new URL("/", request.url));
    }
  } catch {
    // Si le backend est inaccessible, laisser passer (évite la boucle infinie au démarrage)
    return NextResponse.next();
  }

  return NextResponse.next();
}

export const config = {
  // Appliquer le middleware à toutes les routes sauf API et static files
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
