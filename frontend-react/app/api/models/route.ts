import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const API_URL = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function GET() {
  try {
    const res = await fetch(`${API_URL}/models`, {
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) throw new Error(`Backend returned ${res.status}`);
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      {
        openrouter: {
          display_name: "Llama 3.2 3B (OpenRouter)",
          description: "Recommended",
          parameters: "3B",
          requires_gpu: false,
          loaded: true,
        },
      },
      { status: 200 }
    );
  }
}
