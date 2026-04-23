import { NextRequest } from "next/server";

const API_URL = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.API_KEY ?? "";

const headers = () => ({
  "Content-Type": "application/json",
  ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
});

export async function POST() {
  try {
    const res = await fetch(`${API_URL}/sessions`, {
      method: "POST",
      headers: headers(),
      signal: AbortSignal.timeout(15_000),
    });
    const data = await res.json().catch(() => ({}));
    return Response.json(data, { status: res.status });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Upstream sessions service unavailable";
    return Response.json({ error: msg }, { status: 503 });
  }
}

export async function GET(req: NextRequest) {
  const sid = req.nextUrl.searchParams.get("session_id");
  if (!sid) return Response.json({ error: "session_id required" }, { status: 400 });
  try {
    const res = await fetch(`${API_URL}/sessions/${sid}`, {
      headers: headers(),
      signal: AbortSignal.timeout(15_000),
    });
    const data = await res.json().catch(() => ({}));
    return Response.json(data, { status: res.status });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Upstream sessions service unavailable";
    return Response.json({ error: msg }, { status: 503 });
  }
}
