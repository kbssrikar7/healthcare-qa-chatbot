"use client";

import { useExternalStoreRuntime } from "@assistant-ui/react";
import type { AppendMessage, TextMessagePart } from "@assistant-ui/react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { UI_SUGGESTIONS } from "./suggestions";
import type {
  ApiResponse,
  FeedbackState,
  HistoryItem,
  MediMessage,
  Settings,
} from "./types";

// ── Chat context ───────────────────────────────────────────────────────────

interface ChatCtxValue {
  settings: Settings;
  updateSettings: (s: Partial<Settings>) => void;
  newConversation: () => void;
  /** Clears the thread and re-runs the question (same flow as sending from the composer). */
  openHistoryItem: (question: string) => void;
  submitFeedback: (responseId: string, helpful: boolean) => Promise<void>;
  feedbackState: FeedbackState;
  history: HistoryItem[];
  clearHistory: () => void;
  isRunning: boolean;
}

const ChatCtx = createContext<ChatCtxValue>({} as ChatCtxValue);

interface MetaCtxValue {
  getMetadata: (messageId: string) => ApiResponse | undefined;
}
const MetaCtx = createContext<MetaCtxValue>({ getMetadata: () => undefined });

// Runtime context — lets assistant.tsx pull the runtime out
const RuntimeCtx = createContext<ReturnType<typeof useExternalStoreRuntime<MediMessage>> | null>(null);

export function useMediQuery() { return useContext(ChatCtx); }
export function useMessageMetadata(id: string) { return useContext(MetaCtx).getMetadata(id); }
export function useRuntimeFromContext() {
  const r = useContext(RuntimeCtx);
  if (!r) throw new Error("useRuntimeFromContext must be used inside ChatProvider");
  return r;
}

// ── Suggestions ────────────────────────────────────────────────────────────

const SUGGESTIONS = UI_SUGGESTIONS.map(({ prompt, text, description }) => ({
  prompt,
  text,
  description,
}));

// ── Local history ──────────────────────────────────────────────────────────

const HISTORY_KEY = "mq_history";

function loadHistory(): HistoryItem[] {
  if (typeof window === "undefined") return [];
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]"); } catch { return []; }
}

function saveHistory(items: HistoryItem[]) {
  if (typeof window === "undefined") return;
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 20)));
}

// ── Provider ───────────────────────────────────────────────────────────────

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<MediMessage[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [settings, setSettings] = useState<Settings>({
    modelChoice: "ollama",
    pipeline: "standard",
    numSources: 3,
    fullXai: false,
  });
  const [feedbackState, setFeedbackState] = useState<FeedbackState>({});
  // Always start empty so SSR + first client paint match; hydrate from localStorage after mount.
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const sessionIdRef = useRef<string | null>(null);
  const metaMap = useRef(new Map<string, ApiResponse>());
  const requestSeqRef = useRef(0);
  const activeControllerRef = useRef<AbortController | null>(null);

  const submitQuestion = useCallback(async (question: string) => {
    if (!question.trim()) return;
    const requestId = ++requestSeqRef.current;
    activeControllerRef.current?.abort();
    const controller = new AbortController();
    activeControllerRef.current = controller;

    const userMsg: MediMessage = { id: crypto.randomUUID(), role: "user", content: question, timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setIsRunning(true);

    setHistory((prev) => {
      const filtered = prev.filter((h) => h.question !== question);
      const next = [
        { question, timestamp: new Date().toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) },
        ...filtered,
      ].slice(0, 20);
      saveHistory(next);
      return next;
    });

    const s = settingsRef.current;
    try {
      const resp = await fetch("/api/mediquery", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          num_sources: s.numSources,
          model_choice: s.modelChoice !== "auto" ? s.modelChoice : undefined,
          use_langchain: s.pipeline === "langchain",
          use_langgraph: s.pipeline === "langgraph",
          session_id: sessionIdRef.current,
          // Full explainability: richer retrieval + token budget (see API low-latency profile).
          low_latency: !s.fullXai,
          include_explanation: s.fullXai,
        }),
        signal: controller.signal,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err?.error ?? `HTTP ${resp.status}`);
      }

      const data: ApiResponse = await resp.json();
      if (requestId !== requestSeqRef.current) return;
      if (data.session_id) sessionIdRef.current = data.session_id;

      const assistantId = crypto.randomUUID();
      metaMap.current.set(assistantId, data);

      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", content: data.answer || "No answer generated.", timestamp: Date.now(), responseId: data.response_id },
      ]);
    } catch (err: unknown) {
      if (controller.signal.aborted || (err instanceof Error && err.name === "AbortError")) return;
      if (requestId !== requestSeqRef.current) return;
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "assistant", content: `Connection error: **${msg}**\n\nThe AI service may be starting up or experiencing high traffic. Please try again in 30 seconds.`, timestamp: Date.now(), isError: true },
      ]);
    } finally {
      if (requestId === requestSeqRef.current) {
        setIsRunning(false);
        activeControllerRef.current = null;
      }
    }
  }, []);

  const onNew = useCallback(async (msg: AppendMessage) => {
    const question = (msg.content as TextMessagePart[])
      .filter((p) => p.type === "text")
      .map((p) => p.text)
      .join(" ")
      .trim();
    await submitQuestion(question);
  }, [submitQuestion]);

  const runtime = useExternalStoreRuntime<MediMessage>({
    messages,
    isRunning,
    onNew,
    convertMessage: (msg) => ({
      role: msg.role,
      content: [{ type: "text", text: msg.content }],
      id: msg.id,
    }),
    suggestions: SUGGESTIONS,
  });

  const newConversation = useCallback(() => {
    requestSeqRef.current += 1;
    if (activeControllerRef.current) {
      activeControllerRef.current.abort();
      activeControllerRef.current = null;
    }
    setMessages([]);
    setIsRunning(false);
    setFeedbackState({});
    sessionIdRef.current = null;
    metaMap.current = new Map();
  }, []);

  const openHistoryItem = useCallback(
    (question: string) => {
      const q = question.trim();
      if (!q || isRunning) return;
      newConversation();
      void submitQuestion(q);
    },
    [isRunning, newConversation, submitQuestion],
  );

  const updateSettings = useCallback((s: Partial<Settings>) => {
    setSettings((prev) => ({ ...prev, ...s }));
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    if (typeof window !== "undefined") localStorage.removeItem(HISTORY_KEY);
  }, []);

  const submitFeedback = useCallback(async (responseId: string, helpful: boolean) => {
    try {
      const resp = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          response_id: responseId,
          rating: helpful ? 5 : 1,
          was_helpful: helpful,
          was_accurate: helpful,
          was_safe: true,
          session_id: sessionIdRef.current,
        }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err?.error ?? `Feedback request failed (${resp.status})`);
      }
      setFeedbackState((prev) => ({ ...prev, [responseId]: helpful ? "Accurate" : "Inaccurate" }));
    } catch (e) { console.error("Feedback failed:", e); }
  }, []);

  return (
    <ChatCtx.Provider value={{ settings, updateSettings, newConversation, openHistoryItem, submitFeedback, feedbackState, history, clearHistory, isRunning }}>
      <MetaCtx.Provider value={{ getMetadata: (id) => metaMap.current.get(id) }}>
        <RuntimeCtx.Provider value={runtime}>
          {children}
        </RuntimeCtx.Provider>
      </MetaCtx.Provider>
    </ChatCtx.Provider>
  );
}
