"use client";

import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { useMediQuery } from "./chat-provider";
import type { Pipeline } from "./types";

interface ModelInfo {
  display_name: string;
  description: string;
  parameters: string;
  requires_gpu: boolean;
  loaded: boolean;
}

async function fetchModels(): Promise<Record<string, ModelInfo>> {
  try {
    const res = await fetch("/api/models", { signal: AbortSignal.timeout(10_000) });
    if (res.ok) return res.json();
  } catch { /* ignore */ }
  return {
    ollama: { display_name: "Llama 3.2 3B (Local)", description: "Local Ollama", parameters: "3B", requires_gpu: false, loaded: true },
    openrouter: { display_name: "Llama 3.2 3B (OpenRouter)", description: "Via API", parameters: "3B", requires_gpu: false, loaded: false },
  };
}

async function clearCache(): Promise<"cleared" | "empty" | "error"> {
  try {
    const res = await fetch("/api/clear-cache", { method: "POST", signal: AbortSignal.timeout(30_000) });
    if (!res.ok) return "error";
    const data = await res.json().catch(() => ({}));
    return data.cleared ? "cleared" : "empty";
  } catch { return "error"; }
}

const PIPELINES: { value: Pipeline; label: string; desc: string }[] = [
  { value: "standard",  label: "Standard",        desc: "Recommended" },
  { value: "langchain", label: "LangChain LCEL",  desc: "Chain-of-thought" },
  { value: "langgraph", label: "LangGraph",        desc: "Self-correcting" },
];

export function Sidebar() {
  const { settings, updateSettings, newConversation, openHistoryItem, history, clearHistory, isRunning } = useMediQuery();
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [cacheStatus, setCacheStatus] = useState<"idle" | "cleared" | "empty" | "error">("idle");

  useEffect(() => { fetchModels().then(setModels); }, []);

  const modelKeys = Object.keys(models);
  const selectedPipeline = PIPELINES.find((p) => p.value === settings.pipeline);

  const handleClearCache = async () => {
    const result = await clearCache();
    if (result === "cleared") clearHistory();
    setCacheStatus(result);
    setTimeout(() => setCacheStatus("idle"), 3000);
  };

  return (
    <div className="flex h-full w-full flex-col gap-2 overflow-y-auto px-3 py-2">

      {/* ── Brand ── */}
      <div className="flex items-center gap-2.5 px-0 pb-1">
        <span className="inline-flex size-8 items-center justify-center">
          <img src="/logo.svg" alt="MediQuery" width={22} height={22} />
        </span>
        <p className="text-sm font-semibold leading-none tracking-tight">MediQuery</p>
      </div>

      {/* ── New conversation ── */}
      <button
        onClick={newConversation}
        disabled={isRunning}
        className="flex w-full items-center gap-2.5 rounded-md px-1 py-2 text-sm font-medium transition-colors hover:bg-accent/50 hover:text-accent-foreground disabled:opacity-40"
      >
        <span className="inline-flex size-5 items-center justify-center rounded-full border border-border/80 bg-background">
          <svg className="size-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
          </svg>
        </span>
        New chat
      </button>

      {/* ── Model ── */}
      <Section label="Model">
          {modelKeys.length > 1 ? (
            <div className="relative w-full">
              <select
                value={settings.modelChoice}
                onChange={(e) => updateSettings({ modelChoice: e.target.value })}
                className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-2 pr-8 text-sm leading-5 text-zinc-100 shadow-sm transition-colors hover:bg-zinc-800 focus:outline-none focus:ring-2 focus:ring-zinc-500/30 [color-scheme:dark]"
              >
                {modelKeys.map((k) => (
                  <option key={k} value={k} className="bg-zinc-900 text-zinc-100">
                    {models[k].display_name} ({models[k].parameters})
                  </option>
                ))}
              </select>
              <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground">
                <svg className="size-3.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path
                    fillRule="evenodd"
                    d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.168l3.71-3.94a.75.75 0 1 1 1.08 1.04l-4.25 4.5a.75.75 0 0 1-1.08 0l-4.25-4.5a.75.75 0 0 1 .02-1.06Z"
                    clipRule="evenodd"
                  />
                </svg>
              </span>
            </div>
          ) : (
            <p className="px-1 text-sm text-muted-foreground">
              {models[modelKeys[0]]?.display_name ?? "TinyLlama 1.1B"} · {models[modelKeys[0]]?.parameters ?? "1.1B"}
            </p>
          )}
      </Section>

      {/* ── Pipeline ── */}
      <Section label="Pipeline">
          <div className="flex flex-col gap-1.5">
            <div className="relative w-full">
              <select
                value={settings.pipeline}
                onChange={(e) => updateSettings({ pipeline: e.target.value as Pipeline })}
                className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-2 pr-8 text-sm font-medium leading-5 text-zinc-100 shadow-sm transition-colors hover:bg-zinc-800 focus:outline-none focus:ring-2 focus:ring-zinc-500/30 [color-scheme:dark]"
              >
                {PIPELINES.map((p) => (
                  <option key={p.value} value={p.value} className="bg-zinc-900 text-zinc-100">
                    {p.label}
                  </option>
                ))}
              </select>
              <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground">
                <svg className="size-3.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path
                    fillRule="evenodd"
                    d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.168l3.71-3.94a.75.75 0 1 1 1.08 1.04l-4.25 4.5a.75.75 0 0 1-1.08 0l-4.25-4.5a.75.75 0 0 1 .02-1.06Z"
                    clipRule="evenodd"
                  />
                </svg>
              </span>
            </div>
            <p className="px-1 text-xs leading-relaxed text-muted-foreground">
              {selectedPipeline?.value === "standard" && "Balanced default pipeline for most questions."}
              {selectedPipeline?.value === "langchain" && "Chain-based workflow with richer reasoning flow."}
              {selectedPipeline?.value === "langgraph" && "Graph workflow with iterative self-correction steps."}
            </p>
          </div>
      </Section>

      {/* ── Sources slider ── */}
      <div className="px-1 py-1">
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Number of references</span>
            <span className="inline-flex min-w-8 items-center justify-center rounded border border-border bg-background px-2 py-0.5 text-xs font-medium text-foreground">
              {settings.numSources}
            </span>
          </div>
          <input
            type="range"
            min={2}
            max={10}
            value={settings.numSources}
            onChange={(e) => updateSettings({ numSources: parseInt(e.target.value) })}
            className="w-full accent-primary"
          />
          <p className="text-xs leading-relaxed text-muted-foreground">
            More references can improve grounding, but usually increase response time.
          </p>
        </div>
      </div>

      {/* ── Full XAI toggle ── */}
      <div className="px-1 py-1">
        <button
          type="button"
          role="switch"
          aria-checked={settings.fullXai}
          onClick={() => updateSettings({ fullXai: !settings.fullXai })}
          className={`flex w-full cursor-pointer items-center justify-between rounded-md px-1 py-1.5 text-sm transition-colors hover:bg-accent/60 ${
            settings.fullXai ? "bg-accent/70" : ""
          }`}
        >
          <span className="flex flex-col items-start gap-0.5 text-left">
            <span className="font-medium">Full explainability</span>
            <span className="text-xs text-muted-foreground">Enable NLI checks, rerank, and XAI (adds latency)</span>
          </span>
          <div
            className={`relative ml-3 h-6 w-14 shrink-0 rounded-full border transition-colors ${
              settings.fullXai
                ? "border-primary bg-primary/90"
                : "border-border bg-muted-foreground/20"
            }`}
          >
            <span
              className={`absolute inset-y-0 flex items-center text-[9px] font-semibold uppercase tracking-wide ${
                settings.fullXai
                  ? "left-2 px-1 text-primary-foreground"
                  : "right-2 px-1 text-muted-foreground"
              }`}
            >
              {settings.fullXai ? "On" : "Off"}
            </span>
            <div
              className={`absolute top-1/2 h-5 w-5 -translate-y-1/2 rounded-full border border-zinc-500 bg-zinc-100 shadow-sm transition-transform ${
                settings.fullXai ? "translate-x-8" : "translate-x-0.5"
              }`}
            />
          </div>
        </button>
      </div>

      {/* ── History ── */}
      <Section label="Recent Questions" className="pt-1">
        {history.length === 0 ? (
          <p className="px-1 text-xs text-muted-foreground">No questions yet.</p>
        ) : (
          <div className="flex flex-col gap-0.5">
            {history.slice(0, 10).map((item) => (
              <button
                key={`${item.timestamp}::${item.question}`}
                type="button"
                title={item.question}
                disabled={isRunning}
                onClick={() => openHistoryItem(item.question)}
                className="flex w-full cursor-pointer flex-col gap-0.5 rounded-md px-2.5 py-1.5 text-left transition-colors hover:bg-accent/60 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <span className="truncate text-xs font-medium">
                  {item.question.length > 52 ? item.question.slice(0, 52) + "…" : item.question}
                </span>
                <span className="text-[10px] text-muted-foreground">{item.timestamp}</span>
              </button>
            ))}
          </div>
        )}
      </Section>

      <div className="flex-1 min-h-0" />
      <div className="px-1 pb-1 pt-3">
        <button
          type="button"
          onClick={handleClearCache}
          className={cn(
            "group flex w-full items-center justify-center gap-2 rounded-xl px-3 py-2.5 text-xs font-medium shadow-sm transition-all active:scale-[0.99]",
            cacheStatus === "idle" &&
              "border border-zinc-700/50 bg-zinc-800/50 text-zinc-200 hover:border-zinc-600 hover:bg-zinc-800 hover:text-zinc-50",
            cacheStatus === "cleared" &&
              "border border-emerald-500/40 bg-emerald-950/40 text-emerald-300 hover:border-emerald-500/50 hover:bg-emerald-950/60 hover:text-emerald-200",
            cacheStatus === "empty" &&
              "border border-zinc-600/50 bg-zinc-800/50 text-zinc-400 hover:border-zinc-500 hover:bg-zinc-800",
            cacheStatus === "error" &&
              "border border-red-500/40 bg-red-950/30 text-red-300 hover:border-red-500/50 hover:bg-red-950/45 hover:text-red-200",
          )}
        >
          <RefreshCw
            className={cn(
              "size-3.5 shrink-0 opacity-80 transition-transform group-hover:rotate-[-25deg]",
              cacheStatus === "cleared" && "text-emerald-400",
              cacheStatus === "error" && "text-red-400",
              (cacheStatus === "idle" || cacheStatus === "empty") && "text-zinc-400",
            )}
            aria-hidden
          />
          <span>
            {cacheStatus === "cleared"
              ? "Cache cleared"
              : cacheStatus === "empty"
              ? "Nothing to clear"
              : cacheStatus === "error"
              ? "Could not clear cache"
              : "Clear response cache"}
          </span>
        </button>
      </div>
    </div>
  );
}

function Section({
  label,
  children,
  className = "",
}: {
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`flex flex-col gap-1 px-1 py-1 ${className}`}>
      <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground px-1">{label}</p>
      <div>{children}</div>
    </div>
  );
}
