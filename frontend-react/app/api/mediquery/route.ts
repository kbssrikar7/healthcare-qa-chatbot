import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const API_URL = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.API_KEY || "";

export async function POST(req: NextRequest) {
  const body = await req.json();
  try {
    const res = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
      },
      body: JSON.stringify(body),
      cache: "no-store",
      signal: AbortSignal.timeout(1_800_000),
    });
    const data = await res.json();
    return Response.json(data, { status: res.status });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Upstream error";
    return Response.json({ error: msg }, { status: 503 });
  }
}
