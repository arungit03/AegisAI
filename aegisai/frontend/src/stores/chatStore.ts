/** Chat state management using Zustand */
import { create } from 'zustand';
import type { Conversation, Message, SourceCitation } from '../types';

interface ChatState {
  conversations: Conversation[];
  currentConversation: Conversation | null;
  messages: Message[];
  isLoading: boolean;
  isStreaming: boolean;
  error: string | null;
  // Streaming state
  streamingMessageId: string | null;
  streamingContent: string;
  setConversations: (conversations: Conversation[]) => void;
  addConversation: (conversation: Conversation) => void;
  updateConversation: (id: string, data: Partial<Conversation>) => void;
  removeConversation: (id: string) => void;
  setCurrentConversation: (conversation: Conversation | null) => void;
  setMessages: (messages: Message[]) => void;
  addMessage: (message: Message) => void;
  updateLastMessage: (content: string, sources?: SourceCitation[]) => void;
  setIsLoading: (loading: boolean) => void;
  setIsStreaming: (streaming: boolean) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
  reset: () => void;
  // Streaming methods
  startStreamingMessage: (tempId: string) => void;
  appendStreamingToken: (token: string) => void;
  completeStreamingMessage: (message: Message) => void;
  cancelStreamingMessage: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  conversations: [],
  currentConversation: null,
  messages: [],
  isLoading: false,
  isStreaming: false,
  error: null,
  streamingMessageId: null,
  streamingContent: '',

  setConversations: (conversations) => set({ conversations }),
  addConversation: (conversation) => set((state) => ({
    conversations: [conversation, ...state.conversations],
  })),
  updateConversation: (id, data) => set((state) => ({
    conversations: state.conversations.map((c) =>
      c.id === id ? { ...c, ...data } : c
    ),
    currentConversation: state.currentConversation?.id === id
      ? { ...state.currentConversation, ...data }
      : state.currentConversation,
  })),
  removeConversation: (id) => set((state) => ({
    conversations: state.conversations.filter((c) => c.id !== id),
    currentConversation: state.currentConversation?.id === id ? null : state.currentConversation,
  })),
  setCurrentConversation: (conversation) => set({ currentConversation: conversation }),
  setMessages: (messages) => set({ messages }),
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message],
  })),
  updateLastMessage: (content, sources) => set((state) => ({
    messages: state.messages.map((m, i) =>
      i === state.messages.length - 1 && m.role === 'assistant'
        ? { ...m, content, sources: sources || m.sources }
        : m
    ),
  })),
  setIsLoading: (isLoading) => set({ isLoading }),
  setIsStreaming: (isStreaming) => set({ isStreaming }),
  setError: (error) => set({ error }),
  clearError: () => set({ error: null }),
  reset: () => set({
    conversations: [],
    currentConversation: null,
    messages: [],
    isLoading: false,
    isStreaming: false,
    error: null,
    streamingMessageId: null,
    streamingContent: '',
  }),
  // Streaming methods
  startStreamingMessage: (tempId: string) => set({
    isStreaming: true,
    streamingMessageId: tempId,
    streamingContent: '',
  }),
  appendStreamingToken: (token: string) => set((state) => ({
    streamingContent: state.streamingContent + token,
  })),
  completeStreamingMessage: (message: Message) => set((state) => ({
    messages: state.messages.map(m => m.id === state.streamingMessageId ? message : m),
    isStreaming: false,
    streamingMessageId: null,
    streamingContent: '',
  })),
  cancelStreamingMessage: () => set((state) => ({
    messages: state.messages.filter(m => m.id !== state.streamingMessageId),
    isStreaming: false,
    streamingMessageId: null,
    streamingContent: '',
  })),
}));