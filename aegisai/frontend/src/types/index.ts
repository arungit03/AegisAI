/** TypeScript types for the frontend */

export interface User {
  id: string;
  email: string;
  username: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  last_login: string | null;
  created_at: string;
  updated_at: string;
  role: Role;
  department: Department | null;
}

export interface Role {
  id: string;
  name: string;
  description: string | null;
  permissions: string | null;
  created_at: string;
}

export interface Department {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface Token {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface Document {
  id: string;
  filename: string;
  original_filename: string;
  file_size: number;
  mime_type: string;
  extension: string;
  title: string | null;
  description: string | null;
  classification: DocumentClassification;
  department: string | null;
  tags: string[] | null;
  version: number;
  status: DocumentStatus;
  page_count: number | null;
  chunk_count: number | null;
  error_message: string | null;
  processed_at: string | null;
  uploaded_by_id: string;
  department_id: string | null;
  created_at: string;
  updated_at: string;
  permissions: DocumentPermission[];
}

export type DocumentClassification =
  | 'public_internal'
  | 'confidential'
  | 'restricted'
  | 'highly_restricted';

export type DocumentStatus =
  | 'uploaded'
  | 'processing'
  | 'processed'
  | 'failed'
  | 'deleted';

export interface DocumentPermission {
  id: string;
  document_id: string;
  role_id: string | null;
  user_id: string | null;
  can_read: boolean;
  can_write: boolean;
  can_delete: boolean;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  user_id: string;
  is_archived: boolean;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: MessageRole;
  content: string;
  sources: SourceCitation[] | null;
  token_count: number | null;
  model_used: string | null;
  processing_time_ms: number | null;
  created_at: string;
}

export type MessageRole = 'user' | 'assistant' | 'system';

export interface SourceCitation {
  document_id: string;
  document_filename: string;
  document_title: string | null;
  page_number: number | null;
  chunk_text: string;
  similarity_score: number;
  // Enhanced citation metadata
  section_title: string | null;
  chunk_index: number | null;
  char_start: number | null;
  char_end: number | null;
  author: string | null;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  enable_hybrid?: boolean;
}

export interface ChatResponse {
  message: string;
  conversation_id: string;
  sources: SourceCitation[];
  token_count: number | null;
  processing_time_ms: number | null;
  confidence?: number | null;
  pipeline?: string | null;
}

/** Paginated response that works with backend's varied field names.
 *  Backend uses: documents, conversations, users, logs (not "items").
 *  This type accepts both for backward compatibility / safety.
 */
export interface PaginatedResponse<T> {
  items?: T[];
  documents?: T[];
  conversations?: T[];
  users?: T[];
  logs?: T[];
  total: number;
  page: number;
  page_size: number;
}

/** Helper to extract the array from any PaginatedResponse shape. */
export function getPageItems<T>(response: PaginatedResponse<T>): T[] {
  return response.items
    ?? response.documents
    ?? response.conversations
    ?? response.users
    ?? response.logs
    ?? [];
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface ServiceStatus {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency_ms: number | null;
  details: Record<string, unknown> | null;
}

export interface SystemStatusResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  services: Record<string, ServiceStatus>;
  uptime_seconds: number;
}

export interface AuditLog {
  id: string;
  action: string;
  user_id: string | null;
  ip_address: string | null;
  user_agent: string | null;
  request_id: string | null;
  resource_type: string | null;
  resource_id: string | null;
  details: Record<string, unknown> | null;
  success: boolean;
  error_message: string | null;
  created_at: string;
}