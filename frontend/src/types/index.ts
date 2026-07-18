export interface Citation {
  chunk_id: string;
  title: string;
  source: string;
  page: number;
  passage: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  confidence: number;
  retrieval_quality: "GOOD" | "PARTIAL" | "BAD";
  retrieval_reason: string;
  query_rewritten: boolean;
  web_search_used: boolean;
  session_id: string;
}

export interface QueryRequest {
  query: string;
  session_id?: string;
  mode?: "rag" | "general_chat" | "hybrid";
  memory_type?: "buffer" | "summary" | "token";
}

export interface IngestionResponse {
  status: string;
  document_id: string;
  filename: string;
  chunks_created: number;
  message: string;
}

export interface HealthResponse {
  status: string;
  pipeline_ready: boolean;
  message: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp?: number;
}
