const API_URL = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY_ADMIN = process.env.API_KEY_ADMIN ?? process.env.API_KEY ?? "";

export async function POST() {
  try {
    const res = await fetch(`${API_URL}/clear-cache`, {
      method: "POST",
      headers: API_KEY_ADMIN ? { "X-API-Key": API_KEY_ADMIN } : {},
      signal: AbortSignal.timeout(30_000),
    });
    const data = await res.json().catch(() => ({}));
    // Forward cleared flag from backend so the UI can show accurate feedback
    return Response.json({ ok: res.ok, cleared: data.cleared ?? false }, { status: res.status });
  } catch {
    return Response.json({ ok: false, cleared: false }, { status: 503 });
  }
}
