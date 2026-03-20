"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface CurrentUser {
  id: string;
  email: string;
  role: string;
  totp_enabled: boolean;
}

export function useCurrentUser() {
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/auth/users/me`, { credentials: "include" })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setUser(data))
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  return { user, loading };
}

export async function logout(): Promise<void> {
  await fetch(`${API_URL}/auth/jwt/logout`, {
    method: "POST",
    credentials: "include",
  });
}
