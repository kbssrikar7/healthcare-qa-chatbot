"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { ApiResponse } from "./types";
import { useMediQuery } from "./chat-provider";

// ── Collapsible section ────────────────────────────────────────────────────

function Section({
  title,
  count,
  children,
  defaultOpen = false,
}: {
  title: string;
  count?: number;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="overflow-hidden rounded-lg border border-zinc-800/90 bg-zinc-950/40">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-800/60"
      >
        {open ? <ChevronDown className="size-3.5 shrink-0 text-zinc-500" /> : <ChevronRight className="size-3.5 shrink-0 text-zinc-500" />}
        <span>{title}</span>
        {count !== undefined && (
          <span className="ml-auto rounded-md bg-zinc-800 px-2 py-0.5 text-xs tabular-nums text-zinc-400">{count}</span>
        )}
      </button>
      {open && <div className="border-t border-zinc-800/80 bg-zinc-950/30 px-3 py-3">{children}</div>}
    </div>
  );
}

// ── Confidence bar ─────────────────────────────────────────────────────────

/** True only when the backend explicitly says scoring is missing (avoid matching normal prose). */
function confidenceUnavailable(explanation?: string): boolean {
  const t = (explanation ?? "").trim().toLowerCase();
  if (!t) return false;
  return (
    /^n\/a$|^not available\.?$|^unavailable\.?$|^no scoring\.?$/i.test(t) ||
    /confidence (could not be|was not) (classified|computed)/i.test(t) ||
    /^unable to (compute|calculate|determine) confidence/i.test(t)
  );
}

/** Same pattern as sidebar “Number of references”: label, value chip, range, helper. Read-only. */
function ReadonlyMetricSlider({
  label,
  valuePercent,
  helper,
}: {
  label: string;
  valuePercent: number;
  helper?: string;
}) {
  const v = Math.min(100, Math.max(0, Math.round(Number.isFinite(valuePercent) ? valuePercent : 0)));
  return (
    <div className="px-1 py-1">
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">{label}</span>
          <span className="inline-flex min-w-8 items-center justify-center rounded border border-border bg-background px-2 py-0.5 text-xs font-medium text-foreground">
            {v}%
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={100}
          value={v}
          readOnly
          tabIndex={-1}
          className="w-full accent-primary pointer-events-none"
          aria-hidden
        />
        {helper ? <p className="text-xs leading-relaxed text-muted-foreground">{helper}</p> : null}
      </div>
    </div>
  );
}

function ConfidenceBar({ score, level, explanation }: { score: number; level: string; explanation?: string }) {
  if (level === "unknown" && (!explanation || confidenceUnavailable(explanation))) {
    return (
      <div className="px-1 py-1">
        <p className="text-xs leading-relaxed text-muted-foreground">Confidence could not be classified for this response.</p>
      </div>
    );
  }

  if (confidenceUnavailable(explanation)) {
    return (
      <div className="px-1 py-1">
        <p className="text-xs leading-relaxed text-muted-foreground">
          {explanation?.trim() || "Confidence details were not computed for this response."}
        </p>
      </div>
    );
  }

  const pctFloat = Math.min(1, Math.max(0, Number.isFinite(score) ? score : 0)) * 100;
  const explText = explanation && !confidenceUnavailable(explanation) ? explanation.trim() : "";
  // If explanation already starts with the level word, don't prepend a separate label.
  const levelWord = level && level !== "unknown" ? level.toLowerCase() : "";
  const alreadyLabeled = levelWord && explText.toLowerCase().startsWith(levelWord);
  const levelLine = !alreadyLabeled && levelWord
    ? `${levelWord.charAt(0).toUpperCase() + levelWord.slice(1)} confidence.`
    : "";
  const helper = [levelLine, explText]
    .filter(Boolean)
    .join(" ")
    .trim();

  return <ReadonlyMetricSlider label="Confidence" valuePercent={pctFloat} helper={helper || undefined} />;
}

// ── XAI breakdown ──────────────────────────────────────────────────────────

const BREAKDOWN_KEYS = [
  { key: "retrieval_confidence",    label: "Retrieval Quality",    weight: "retrieval" },
  { key: "generation_confidence",   label: "Generation Certainty", weight: "generation" },
  { key: "consistency_score",       label: "Self-Consistency",     weight: "consistency" },
  { key: "source_agreement",        label: "Source Agreement",     weight: "source_agreement" },
  { key: "medical_entity_coverage", label: "Entity Coverage",      weight: "entity_coverage" },
] as const;

function BreakdownPanel({ breakdown }: { breakdown: NonNullable<ApiResponse["confidence_breakdown"]> }) {
  return (
    <div className="flex flex-col gap-1">
      {BREAKDOWN_KEYS.map(({ key, label, weight }) => {
        const raw = breakdown[key as keyof typeof breakdown];
        const val = typeof raw === "number" ? raw : Number(raw);
        const pct = Number.isFinite(val) ? val * 100 : 0;
        const wRaw = breakdown.signal_weights?.[weight];
        const wPct = typeof wRaw === "number" && Number.isFinite(wRaw) ? Math.round(wRaw * 100) : null;
        const helper =
          wPct !== null && wPct > 0
            ? `Blend weight ${wPct}% in the confidence model.`
            : undefined;
        return <ReadonlyMetricSlider key={key} label={label} valuePercent={pct} helper={helper} />;
      })}
    </div>
  );
}

// ── Sources ────────────────────────────────────────────────────────────────

function SourceItem({ src, index }: { src: NonNullable<ApiResponse["sources"]>[number]; index: number }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-zinc-800/90 bg-zinc-950/50 p-3">
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs text-zinc-500">#{index + 1}</span>
        <span className="rounded-md bg-zinc-800 px-2 py-0.5 text-xs font-medium text-zinc-300">
          {Math.round(src.score * 100)}% match
        </span>
      </div>
      <p className="text-xs leading-relaxed text-zinc-400">
        {expanded ? src.content : src.content.slice(0, 180) + (src.content.length > 180 ? "…" : "")}
      </p>
      <div className="flex items-center justify-between">
        <span className="max-w-[60%] truncate text-[10px] text-zinc-500">{src.source}</span>
        <div className="flex gap-1.5">
          <button type="button" onClick={() => setExpanded((e) => !e)} className="text-[10px] text-zinc-400 underline-offset-2 hover:text-zinc-200 hover:underline">
            {expanded ? "Collapse" : "Full text"}
          </button>
          {src.url && <a href={src.url} target="_blank" rel="noopener noreferrer" className="text-[10px] text-zinc-400 underline-offset-2 hover:text-zinc-200 hover:underline">Source</a>}
        </div>
      </div>
    </div>
  );
}

// ── Hallucination ──────────────────────────────────────────────────────────

function HallucinationPanel({ hal }: { hal: NonNullable<ApiResponse["hallucination"]> }) {
  const pct = Math.round(hal.score * 100);
  const clean = !hal.has_hallucination;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-3">
        <span className={`rounded-md px-2 py-0.5 text-xs font-semibold ${clean ? "bg-emerald-950 text-emerald-300 ring-1 ring-emerald-800/60" : "bg-red-950 text-red-300 ring-1 ring-red-900/60"}`}>
          {clean ? "Clean" : `${hal.type}`}
        </span>
        <span className="text-xs text-muted-foreground">
          Risk: <strong>{pct}%</strong>
        </span>
      </div>
      {hal.explanation && <p className="text-xs leading-relaxed text-zinc-400">{hal.explanation}</p>}
      {hal.medical_accuracy_flags?.length ? (
        <div className="flex flex-wrap gap-1.5">
          {hal.medical_accuracy_flags.map((f, i) => (
            <span key={i} className="rounded-md bg-amber-950/80 px-2 py-0.5 text-[10px] text-amber-200 ring-1 ring-amber-900/50">{f}</span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

// ── Feedback ───────────────────────────────────────────────────────────────

function FeedbackRow({ responseId }: { responseId: string }) {
  const { feedbackState, submitFeedback } = useMediQuery();
  const submitted = feedbackState[responseId];

  if (submitted) {
    return (
      <p className="text-xs text-zinc-500">
        Thanks — marked as <span className="font-medium text-emerald-400">{submitted}</span>.
      </p>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs text-zinc-500">Was this helpful?</span>
      <button
        type="button"
        onClick={() => submitFeedback(responseId, true)}
        className="rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-200 transition-colors hover:border-emerald-700 hover:bg-emerald-950/40"
      >
        Yes
      </button>
      <button
        type="button"
        onClick={() => submitFeedback(responseId, false)}
        className="rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-200 transition-colors hover:border-red-900 hover:bg-red-950/30"
      >
        No
      </button>
    </div>
  );
}

// ── Main AnswerCard ────────────────────────────────────────────────────────

export function AnswerCard({ response }: { response: ApiResponse }) {
  const safety = response.safety;
  const conf = response.confidence;
  const sources = response.sources ?? [];
  const attributions = response.attributions ?? [];
  const verifiedCount = attributions.filter((a) => a.source !== "Unsupported").length;

  const metaParts: string[] = [];
  if (response.model_used)   metaParts.push(response.model_used);
  if (response.pipeline_used) metaParts.push(response.pipeline_used);
  if (response.latency_ms)   metaParts.push(`${Math.round(response.latency_ms)} ms`);
  else if (response.elapsed_time) metaParts.push(`${response.elapsed_time.toFixed(1)} s`);
  if (response.from_cache)   metaParts.push("cached");

  return (
    <div className="flex flex-col gap-4">

      {/* Emergency */}
      {safety?.is_emergency && (
        <div className="rounded-lg border border-red-900/60 bg-red-950/40 px-3 py-2.5 text-sm font-medium text-red-200">
          {safety.emergency_message ?? "Please seek immediate emergency medical attention."}
        </div>
      )}

      {/* Drug warnings */}
      {safety?.drug_warnings?.map((w, i) => (
        <div key={i} className="rounded-lg border border-amber-900/50 bg-amber-950/30 px-3 py-2 text-xs text-amber-100">
          {w}
        </div>
      ))}

      {/* Meta line */}
      {metaParts.length > 0 && (
        <p className="text-[11px] tabular-nums text-zinc-500">{metaParts.join(" · ")}</p>
      )}

      {/* Confidence bar */}
      {conf && (
        <ConfidenceBar score={conf.score} level={conf.level} explanation={conf.explanation} />
      )}

      {/* Populated when API runs with include_explanation (see chat-provider + /ask). */}
      {response.confidence_breakdown && (
        <Section title="XAI signal breakdown" defaultOpen>
          <BreakdownPanel breakdown={response.confidence_breakdown} />
        </Section>
      )}

      {/* Sources */}
      {sources.length > 0 && (
        <Section title="Sources & References" count={sources.length}>
          <div className="flex flex-col gap-2">
            {sources.map((src, i) => <SourceItem key={i} src={src} index={i} />)}
          </div>
        </Section>
      )}

      {/* Claim attribution — shows all claims including unverified ones */}
      {attributions.length > 0 && (
        <Section title="Claim Verification" count={verifiedCount}>
          <div className="flex flex-col gap-2">
            {attributions.map((a, i) => {
              const unsupported = a.source === "Unsupported";
              return (
                <div
                  key={i}
                  className={`border-l-2 pl-3 ${unsupported ? "border-amber-800/50 opacity-60" : "border-primary/30"}`}
                >
                  <p className="text-xs text-foreground italic">&ldquo;{a.claim}&rdquo;</p>
                  <p className="text-[10px] text-muted-foreground mt-0.5 font-mono">
                    {unsupported
                      ? "⚠ Unverified — no matching source found"
                      : `${a.source} · ${Math.round(a.similarity * 100)}% match`}
                  </p>
                </div>
              );
            })}
            {attributions.length > verifiedCount && (
              <p className="text-[10px] text-amber-500/70 mt-1">
                {attributions.length - verifiedCount} claim{attributions.length - verifiedCount > 1 ? "s" : ""} could not be matched to a source.
              </p>
            )}
          </div>
        </Section>
      )}

      {/* Hallucination — skip for Ollama (DeBERTa NLI unreliable on paraphrased answers) */}
      {response.hallucination && response.model_used !== "ollama" && (
        <Section title="Hallucination Analysis">
          <HallucinationPanel hal={response.hallucination} />
        </Section>
      )}
      {!response.hallucination && response.model_used === "ollama" && response.confidence_breakdown && (
        <Section title="Hallucination Analysis">
          <p className="text-xs text-zinc-400">
            Hallucination detection is not applicable for Qwen2.5-7B — the model paraphrases answers by design and NLI scoring is unreliable on exam-format knowledge base chunks. Source grounding is shown above via Source Agreement ({Math.round((response.confidence_breakdown.source_agreement ?? 0) * 100)}%).
          </p>
        </Section>
      )}

      {/* Disclaimer */}
      <p className="border-t border-zinc-800/80 pt-3 text-[11px] leading-relaxed text-zinc-500">
        {response.disclaimer ?? "For educational use only — not a substitute for professional medical advice. Consult a qualified healthcare provider."}
      </p>

      {/* Feedback */}
      {response.response_id && <FeedbackRow responseId={response.response_id} />}
    </div>
  );
}
