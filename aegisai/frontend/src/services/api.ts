/** API service for communicating with the backend */
import axios, { AxiosError, AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import type { Token, LoginRequest, User, Document, Conversation, Message, ChatRequest, ChatResponse, HealthResponse, SystemStatusResponse, PaginatedResponse, AuditLog, Role, Department, DocumentPermission, SourceCitation } from '../types';

const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || '/api';

class ApiService {
  private client: AxiosInstance;
  private accessToken: string | null = null;
  private refreshToken: string | null = null;
  private isRefreshing = false;
  private failedQueue: Array<{
    resolve: (token: string) => void;
    reject: (error: Error) => void;
  }> = [];

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    // Load tokens from localStorage
    this.accessToken = localStorage.getItem('access_token');
    this.refreshToken = localStorage.getItem('refresh_token');

    // Request interceptor to add auth header
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        if (this.accessToken) {
          config.headers.Authorization = `Bearer ${this.accessToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

        if (error.response?.status === 401 && !originalRequest._retry) {
          if (this.isRefreshing) {
            // Queue the request
            return new Promise((resolve, reject) => {
              this.failedQueue.push({ resolve, reject });
            }).then((token) => {
              originalRequest.headers.Authorization = `Bearer ${token}`;
              return this.client(originalRequest);
            }).catch((err) => {
              return Promise.reject(err);
            });
          }

          originalRequest._retry = true;
          this.isRefreshing = true;

          try {
            const newToken = await this.refreshAccessToken();
            this.processQueue(null, newToken);
            originalRequest.headers.Authorization = `Bearer ${newToken}`;
            return this.client(originalRequest);
          } catch (err) {
            this.processQueue(err as Error, '');
            this.logout();
            // Dispatch a custom event to trigger navigation (works outside React components)
            window.dispatchEvent(new CustomEvent('auth:logout'));
            return Promise.reject(err);
          } finally {
            this.isRefreshing = false;
          }
        }

        return Promise.reject(error);
      }
    );
  }

  private processQueue(error: Error | null, token: string) {
    this.failedQueue.forEach(({ resolve, reject }) => {
      if (error) {
        reject(error);
      } else {
        resolve(token);
      }
    });
    this.failedQueue = [];
  }

  private setTokens(accessToken: string, refreshToken: string) {
    this.accessToken = accessToken;
    this.refreshToken = refreshToken;
    localStorage.setItem('access_token', accessToken);
    localStorage.setItem('refresh_token', refreshToken);
  }

  private clearTokens() {
    this.accessToken = null;
    this.refreshToken = null;
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  }

  // Auth endpoints
  async login(credentials: LoginRequest): Promise<Token> {
    const response = await this.client.post<Token>('/auth/login', {
      ...credentials,
      username: credentials.username.trim(),
    });
    this.setTokens(response.data.access_token, response.data.refresh_token);
    return response.data;
  }

  async refreshAccessToken(): Promise<string> {
    if (!this.refreshToken) throw new Error('No refresh token');
    const response = await this.client.post<Token>('/auth/refresh', { refresh_token: this.refreshToken });
    this.setTokens(response.data.access_token, response.data.refresh_token);
    return response.data.access_token;
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/auth/logout');
    } finally {
      this.clearTokens();
    }
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/auth/me');
    return response.data;
  }

  // Health endpoints
  async healthCheck(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health');
    return response.data;
  }

  async readinessCheck(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health/ready');
    return response.data;
  }

  async systemStatus(): Promise<SystemStatusResponse> {
    const response = await this.client.get<SystemStatusResponse>('/health/status');
    return response.data;
  }

  // User endpoints
  async getUsers(params?: { page?: number; page_size?: number; search?: string; role_id?: string; department_id?: string; is_active?: boolean }): Promise<PaginatedResponse<User>> {
    const response = await this.client.get<PaginatedResponse<User>>('/users', { params });
    return response.data;
  }

  async getUser(userId: string): Promise<User> {
    const response = await this.client.get<User>(`/users/${userId}`);
    return response.data;
  }

  async createUser(data: Partial<User> & { password: string; role_id: string }): Promise<User> {
    const response = await this.client.post<User>('/users', data);
    return response.data;
  }

  async updateUser(userId: string, data: Partial<User>): Promise<User> {
    const response = await this.client.patch<User>(`/users/${userId}`, data);
    return response.data;
  }

  async deleteUser(userId: string): Promise<void> {
    await this.client.delete(`/users/${userId}`);
  }

  // Role endpoints
  async getRoles(): Promise<Role[]> {
    const response = await this.client.get<Role[]>('/users/roles');
    return response.data;
  }

  async createRole(data: { name: string; description?: string; permissions?: string }): Promise<Role> {
    const response = await this.client.post<Role>('/users/roles', data);
    return response.data;
  }

  async updateRole(roleId: string, data: Partial<Role>): Promise<Role> {
    const response = await this.client.patch<Role>(`/users/roles/${roleId}`, data);
    return response.data;
  }

  async deleteRole(roleId: string): Promise<void> {
    await this.client.delete(`/users/roles/${roleId}`);
  }

  // Department endpoints
  async getDepartments(): Promise<Department[]> {
    const response = await this.client.get<Department[]>('/users/departments');
    return response.data;
  }

  async createDepartment(data: { name: string; description?: string }): Promise<Department> {
    const response = await this.client.post<Department>('/users/departments', data);
    return response.data;
  }

  async updateDepartment(deptId: string, data: Partial<Department>): Promise<Department> {
    const response = await this.client.patch<Department>(`/users/departments/${deptId}`, data);
    return response.data;
  }

  async deleteDepartment(deptId: string): Promise<void> {
    await this.client.delete(`/users/departments/${deptId}`);
  }

  // Document endpoints
  async getDocuments(params?: { page?: number; page_size?: number; search?: string; classification?: string; status?: string; department?: string }): Promise<PaginatedResponse<Document>> {
    const response = await this.client.get<PaginatedResponse<Document>>('/documents', { params });
    return response.data;
  }

  async getDocument(documentId: string): Promise<Document> {
    const response = await this.client.get<Document>(`/documents/${documentId}`);
    return response.data;
  }

  async uploadDocument(file: File, metadata: { title?: string; description?: string; classification?: string; department?: string; tags?: string }): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata.title) formData.append('title', metadata.title);
    if (metadata.description) formData.append('description', metadata.description);
    if (metadata.classification) formData.append('classification', metadata.classification);
    if (metadata.department) formData.append('department', metadata.department);
    if (metadata.tags) formData.append('tags', metadata.tags);

    const response = await this.client.post<Document>('/documents', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async updateDocument(documentId: string, data: Partial<Document>): Promise<Document> {
    const response = await this.client.patch<Document>(`/documents/${documentId}`, data);
    return response.data;
  }

  async deleteDocument(documentId: string): Promise<void> {
    await this.client.delete(`/documents/${documentId}`);
  }

  async reindexDocument(documentId: string): Promise<{ message: string; document_id: string }> {
    const response = await this.client.post<{ message: string; document_id: string }>(`/documents/${documentId}/reindex`);
    return response.data;
  }

  // Document permissions
  async getDocumentPermissions(documentId: string): Promise<DocumentPermission[]> {
    const response = await this.client.get<DocumentPermission[]>(`/documents/${documentId}/permissions`);
    return response.data;
  }

  async addDocumentPermission(documentId: string, data: { role_id?: string; user_id?: string; can_read?: boolean; can_write?: boolean; can_delete?: boolean }): Promise<DocumentPermission> {
    const response = await this.client.post<DocumentPermission>(`/documents/${documentId}/permissions`, data);
    return response.data;
  }

  async updateDocumentPermission(documentId: string, permissionId: string, data: Partial<DocumentPermission>): Promise<DocumentPermission> {
    const response = await this.client.patch<DocumentPermission>(`/documents/${documentId}/permissions/${permissionId}`, data);
    return response.data;
  }

  async deleteDocumentPermission(documentId: string, permissionId: string): Promise<void> {
    await this.client.delete(`/documents/${documentId}/permissions/${permissionId}`);
  }

  // Conversation endpoints
  async getConversations(params?: { page?: number; page_size?: number; include_archived?: boolean }): Promise<PaginatedResponse<Conversation>> {
    const response = await this.client.get<PaginatedResponse<Conversation>>('/chat/conversations', { params });
    return response.data;
  }

  async getConversation(conversationId: string): Promise<Conversation> {
    const response = await this.client.get<Conversation>(`/chat/conversations/${conversationId}`);
    return response.data;
  }

  async createConversation(title: string): Promise<Conversation> {
    const response = await this.client.post<Conversation>('/chat/conversations', { title });
    return response.data;
  }

  async updateConversation(conversationId: string, data: Partial<Conversation>): Promise<Conversation> {
    const response = await this.client.patch<Conversation>(`/chat/conversations/${conversationId}`, data);
    return response.data;
  }

  async deleteConversation(conversationId: string): Promise<void> {
    await this.client.delete(`/chat/conversations/${conversationId}`);
  }

  async getMessages(conversationId: string, params?: { page?: number; page_size?: number }): Promise<Message[]> {
    const response = await this.client.get<Message[]>(`/chat/conversations/${conversationId}/messages`, { params });
    return response.data;
  }

  async getConversationMemory(conversationId: string): Promise<Array<{
    id: string;
    summary: string;
    message_count: number;
    token_count: number;
    window_start: string;
    window_end: string;
    created_at: string;
  }>> {
    const response = await this.client.get(`/chat/conversations/${conversationId}/memory`);
    return response.data;
  }

  // Chat endpoint
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await this.client.post<ChatResponse>('/chat', request);
    return response.data;
  }

  // Streaming chat endpoint
  async sendMessageStream(
    request: ChatRequest,
    onToken: (token: string) => void,
    onComplete: (data: { sources: SourceCitation[]; conversation_id: string; processing_time_ms: number; token_count: number | null }) => void
  ): Promise<void> {
    const response = await fetch(`${this.client.defaults.baseURL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.accessToken}`,
      },
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);
            if (data.done) {
              onComplete(data);
            } else if (data.token) {
              onToken(data.token);
            }
          } catch (e) {
            // Ignore parse errors for partial chunks
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // Audit endpoints
  async getAuditLogs(params?: { page?: number; page_size?: number; action?: string; user_id?: string; resource_type?: string; resource_id?: string; success?: boolean; start_date?: string; end_date?: string }): Promise<PaginatedResponse<AuditLog>> {
    const response = await this.client.get<PaginatedResponse<AuditLog>>('/audit-logs', { params });
    return response.data;
  }

  isAuthenticated(): boolean {
    return !!this.accessToken;
  }

  getAccessToken(): string | null {
    return this.accessToken;
  }
}

export const api = new ApiService();
