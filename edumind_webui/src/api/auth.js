import { useUserStore } from "@/stores";

const BASE_URL = (import.meta.env.VITE_EDUMIND_API_BASE_URL ?? "http://localhost:9621").replace(/\/+$/, "");
const API_KEY  = import.meta.env.VITE_LIGHTRAG_API_KEY;
const DEFAULT_WORKSPACE = import.meta.env.VITE_DEFAULT_WORKSPACE ?? "default";

const TOKEN_KEY = "edumind_access_token";
const MODE_KEY  = "edumind_auth_mode";

export function setToken(t) {
  sessionStorage.setItem(TOKEN_KEY, t);
}
export function getToken() {
  return sessionStorage.getItem(TOKEN_KEY) || "";
}
export function clearToken() {
  sessionStorage.removeItem(TOKEN_KEY);
}
function setMode(m) {
  sessionStorage.setItem(MODE_KEY, m);
}
export function getMode(){
  return (sessionStorage.getItem(MODE_KEY)) || null;
}

export function decodeJwt(token) {
  try {
    const [, payloadB64] = token.split(".");
    const json = JSON.parse(atob(payloadB64.replace(/-/g, "+").replace(/_/g, "/")));
    return json ?? null;
  } catch { return null; }
}

function jwtExpired(token, skewSec = 15) {
  const p = decodeJwt(token);
  if (!p?.exp) return false;
  const now = Math.floor(Date.now()/1000);
  return now >= (p.exp - skewSec);
}

function authHeaders(extra) {
  const h = new Headers(extra);
  const t = getToken();
  if (t) h.set("Authorization", `Bearer ${t}`);
  if (API_KEY) h.set("X-API-Key", API_KEY);
  if (!h.has("X-Workspace") && DEFAULT_WORKSPACE) h.set("X-Workspace", DEFAULT_WORKSPACE);
  return h;
}

/** Try to obtain guest token when auth is disabled; otherwise tells us auth is enabled. */
export async function fetchAuthStatus() {
  const resp = await fetch(`${BASE_URL}/auth-status`, { method: "GET" });
  if (!resp.ok) throw new Error(`auth-status failed: ${resp.status}`);
  return resp.json();
}

export function getCurrentUserFromToken() {
  const token = getToken();
  if (!token || jwtExpired(token)) return null;

  const p = decodeJwt(token);
  if (!p || !p.sub) return null;

  const metadata = p.metadata || {};
  const user_id = metadata.user_id;

  return {
    id: user_id,
    username: p.sub,
    full_name: p.sub,
    role: p.role ?? 'user',
    exp: typeof p.exp === 'number' ? p.exp : undefined,
  };
}

/** Public: used by your store.hydrate() */
export async function fetchCurrentUser() {
  // If we already have a valid token, decode user locally.
  const existing = getToken();
  if (existing && !jwtExpired(existing)) {
    const u = getCurrentUserFromToken();
    if (u) return { user: u };
  }

  // Otherwise, see if server has auth disabled and gives us a guest token.
  const status = await fetchAuthStatus();
  if (status?.auth_configured === false && status?.access_token) {
    setToken(status.access_token);
    setMode("disabled");
    const u = getCurrentUserFromToken();
    return { user: u }; // guest user
  }

  // Auth is enabled and no valid token yet
  setMode("enabled");
  return { user: null };
}

/** Public: called by your store.signIn() */
export async function signIn({ username, password }) {
  const form = new URLSearchParams({
    grant_type: "password",
    username,
    password,
    scope: "",
    client_id: "",
    client_secret: "",
  });

  const resp = await fetch(`${BASE_URL}/login`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" }),
    body: form,
  });

  if (!resp.ok) {
    if (resp.status === 401) throw new Error("Incorrect credentials");
    throw new Error(`Login failed: ${resp.status}`);
  }

  const data = await resp.json();
  if (!data?.access_token) throw new Error("No access_token in response");

  setToken(data.access_token);
  setMode(data.auth_mode ?? "enabled");

  // Build user from token (no /me)
  const user = getCurrentUserFromToken();
  return { user };
}

/** Public: used by your store.signOut() */
export async function signOut() {
  clearToken();
  setMode("enabled");
}
