export interface ApiConfidence {
  score: number;
  level: "high" | "medium" | "low" | "unknown";
  explanation: string;
}

export interface ApiConfidenceBreakdown {
  retrieval_confidence: number;
  generation_confidence: number;
  consistency_score: number;
  source_agreement: number;
  medical_entity_coverage: number;
  signal_weights: Record<string, number>;
}

export interface ApiSource {
  content: string;
  source: string;
  score: number;
  url?: string;
}

export interface ApiAttribution {
  claim: string;
  source: string;
  similarity: number;
}

export interface ApiHallucination {
  score: number;
  has_hallucination: boolean;
  type: string;
  explanation: string;
  medical_accuracy_flags?: string[];
}

export interface ApiSafety {
  is_emergency: boolean;
  emergency_message?: string;
  level: "safe" | "caution" | "blocked" | "emergency";
  drug_warnings?: string[];
}

export interface ApiResponse {
  answer: string;
  response_id: string;
  model_used?: string;
  pipeline_used?: string;
  latency_ms?: number;
  elapsed_time?: number;
  session_id?: string;
  confidence?: ApiConfidence;
  confidence_breakdown?: ApiConfidenceBreakdown;
  sources?: ApiSource[];
  attributions?: ApiAttribution[];
  hallucination?: ApiHallucination;
  safety?: ApiSafety;
  rationale?: string;
  disclaimer?: string;
  from_cache?: boolean;
}

export interface MediMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  responseId?: string;
  isError?: boolean;
}

export type Pipeline = "standard" | "langchain" | "langgraph";

export interface Settings {
  modelChoice: string;
  pipeline: Pipeline;
  numSources: number;
  fullXai: boolean;
}

export interface HistoryItem {
  question: string;
  timestamp: string;
}

export interface FeedbackState {
  [responseId: string]: "Accurate" | "Inaccurate";
}
