/** Chat page with ChatGPT-like interface */
import { useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Send,
  Plus,
  Paperclip,
  Settings,
  Copy,
  ThumbsUp,
  ThumbsDown,
  Bot,
  User,
  FileText,
  ChevronLeft,
  ChevronRight,
  MoreVertical,
  MessageSquare,
  Loader2,
  BookOpen,
  Clock,
} from 'lucide-react';
import { format } from 'date-fns';
import { clsx } from 'clsx';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { api } from '../services/api';
import { useChatStore } from '../stores/chatStore';
import { Button } from '../components/ui/Button';
import { Textarea } from '../components/ui/Textarea';
import { SourceCitation, Conversation, Message, getPageItems } from '../types';

const MessageBubble = ({ message, onCopy, onFeedback }: { message: Message; onCopy: (text: string) => void; onFeedback: (messageId: string, helpful: boolean) => void }) => {
  const [showSources, setShowSources] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    onCopy(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isAssistant = message.role === 'assistant';

  return (
    <div className={clsx('flex gap-3', isAssistant ? 'flex-row' : 'flex-row-reverse')}>
      <div
        className={clsx(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium',
          isAssistant ? 'bg-primary-100 text-primary-700' : 'bg-dark-100 text-dark-700'
        )}
      >
        {isAssistant ? <Bot className="h-5 w-5" /> : <User className="h-5 w-5" />}
      </div>
      <div className={clsx('flex-1 max-w-[85%]', isAssistant ? '' : 'text-right')}>
        <div
          className={clsx(
            'inline-block px-4 py-2 rounded-2xl text-sm whitespace-pre-wrap break-words',
            isAssistant
              ? 'bg-white text-dark-900 border border-dark-200 rounded-bl-md'
              : 'bg-primary-600 text-white rounded-br-md'
          )}
        >
          <div className="prose prose-sm max-w-none dark:prose-invert">
            {isAssistant ? (
              <div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(marked.parse(message.content || '') as string) }} />
            ) : (
              message.content
            )}
          </div>
        </div>

        <div className={clsx('flex items-center gap-1 mt-1.5 flex-wrap', isAssistant ? 'justify-start' : 'justify-end')}>
          <span className="text-xs text-dark-400">
            {format(new Date(message.created_at), 'HH:mm')}
          </span>
          {isAssistant && (message as any).confidence !== undefined && (message as any).confidence !== null && (
            <span className="text-xs px-2 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-200" title={`Pipeline: ${(message as any).pipeline || 'legacy'}`}>
              conf {(Number((message as any).confidence) * 100).toFixed(0)}%
            </span>
          )}
          {isAssistant && (message as any).pipeline === 'hybrid' && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-violet-100 text-violet-700 border border-violet-200">hybrid</span>
          )}

          {isAssistant && (
            <>
              <button
                onClick={handleCopy}
                className="p-1 rounded text-dark-400 hover:text-dark-600 hover:bg-dark-100 transition-colors"
                aria-label={copied ? 'Copied!' : 'Copy'}
              >
                {copied ? <span className="text-xs text-green-600">Copied!</span> : <Copy className="h-4 w-4" />}
              </button>
              <button
                onClick={() => onFeedback(message.id, true)}
                className="p-1 rounded text-dark-400 hover:text-green-600 hover:bg-green-50 transition-colors"
                aria-label="Helpful"
              >
                <ThumbsUp className="h-4 w-4" />
              </button>
              <button
                onClick={() => onFeedback(message.id, false)}
                className="p-1 rounded text-dark-400 hover:text-red-600 hover:bg-red-50 transition-colors"
                aria-label="Not helpful"
              >
                <ThumbsDown className="h-4 w-4" />
              </button>
              {message.sources && message.sources.length > 0 && (
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="p-1 rounded text-dark-400 hover:text-primary-600 hover:bg-primary-50 transition-colors"
                  aria-label={showSources ? 'Hide sources' : 'Show sources'}
                >
                  <FileText className="h-4 w-4" />
                </button>
              )}
            </>
          )}
        </div>

        {showSources && message.sources && message.sources.length > 0 && (
          <div className="mt-3 p-3 rounded-lg bg-dark-50 border border-dark-200 animate-in">
            <p className="text-xs font-medium text-dark-700 mb-2">Sources</p>
            <div className="space-y-2">
              {message.sources.map((source: SourceCitation, i: number) => (
                <div key={i} className="text-xs text-dark-600">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-dark-900">{source.document_filename}</span>
                    {source.page_number && <span className="text-dark-500">- Page {source.page_number}</span>}
                    {source.section_title && <span className="text-dark-500">- Section: {source.section_title}</span>}
                    {source.chunk_index !== null && source.chunk_index !== undefined && (
                      <span className="text-dark-500">- Chunk {source.chunk_index + 1}</span>
                    )}
                  </div>
                  {source.document_title && (
                    <div className="text-dark-400 text-xs mb-1">Document: {source.document_title}</div>
                  )}
                  {source.author && (
                    <div className="text-dark-400 text-xs mb-1">Author: {source.author}</div>
                  )}
                  <div className="mt-1 text-dark-500 line-clamp-2">{source.chunk_text}</div>
                  <div className="flex items-center gap-2 mt-1 flex-wrap">
                    <span className="text-xs px-2 py-0.5 rounded bg-primary-100 text-primary-700">
                      {(source.similarity_score * 100).toFixed(0)}% match
                    </span>
                    {source.char_start !== null && source.char_start !== undefined && source.char_end !== null && source.char_end !== undefined && (
                      <span className="text-xs px-2 py-0.5 rounded bg-dark-100 text-dark-600">
                        chars {source.char_start}-{source.char_end}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const Sidebar = ({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  loading,
}: {
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  loading: boolean;
}) => {
  return (
    <aside className="w-72 border-r border-dark-200 bg-white flex flex-col hidden lg:flex">
      <div className="p-4 border-b border-dark-200">
        <Button variant="secondary" className="w-full justify-start gap-2" onClick={onNewConversation}>
          <Plus className="h-4 w-4" />
          New Chat
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="p-4 space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-10 bg-dark-100 rounded animate-pulse" />
            ))}
          </div>
        ) : conversations.length === 0 ? (
          <div className="p-4 text-center text-dark-500">
            <p>No conversations yet</p>
            <p className="text-sm mt-1">Start a new chat to begin</p>
          </div>
        ) : (
          <ul className="divide-y divide-dark-100" role="list">
            {conversations.map((conv) => (
              <li key={conv.id}>
                <button
                  onClick={() => onSelectConversation(conv.id)}
                  className={clsx(
                    'w-full px-3 py-2.5 text-left text-sm transition-colors',
                    currentConversationId === conv.id
                      ? 'bg-primary-50 text-primary-700'
                      : 'text-dark-600 hover:bg-dark-50 hover:text-dark-900',
                    !conv.is_archived ? '' : 'opacity-50'
                  )}
                >
                  <p className="font-medium truncate">{conv.title}</p>
                  <p className="text-xs text-dark-400 mt-0.5 truncate">
                    {format(new Date(conv.updated_at), 'MMM d, HH:mm')}
                  </p>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
};

export const ChatPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const conversationId = searchParams.get('conversation');

  const {
    conversations,
    currentConversation,
    messages,
    isLoading,
    isStreaming,
    streamingMessageId,
    streamingContent,
    setConversations,
    setCurrentConversation,
    setMessages,
    addMessage,
    setIsLoading,
    reset,
    startStreamingMessage,
    appendStreamingToken,
    completeStreamingMessage,
    cancelStreamingMessage,
  } = useChatStore();

  const [newMessage, setNewMessage] = useState('');
  const [showSidebar, setShowSidebar] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [memory, setMemory] = useState<Array<{
    id: string;
    summary: string;
    message_count: number;
    token_count: number;
    window_start: string;
    window_end: string;
    created_at: string;
  }>>([]);
  const [showMemory, setShowMemory] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchConversations = async () => {
      try {
        const response = await api.getConversations({ page_size: 50 });
        setConversations(getPageItems(response));
      } catch (error) {
        console.error('Failed to fetch conversations:', error);
      }
    };
    fetchConversations();
  }, [setConversations]);

  useEffect(() => {
    if (conversationId) {
      const loadConversation = async () => {
        try {
          setIsLoading(true);
          const [conv, msgs] = await Promise.all([
            api.getConversation(conversationId),
            api.getMessages(conversationId),
          ]);
          setCurrentConversation(conv);
          // The API returns messages in chronological order (oldest first).
          // Keep that same order so loaded history and newly appended messages
          // follow one consistent rendering path.
          setMessages(msgs);
        } catch (error) {
          console.error('Failed to load conversation:', error);
        } finally {
          setIsLoading(false);
        }
      };
      loadConversation();

      // Load conversation memory
      const loadMemory = async () => {
        try {
          const memoryData = await api.getConversationMemory(conversationId);
          setMemory(memoryData);
        } catch (error) {
          console.error('Failed to load conversation memory:', error);
        }
      };
      loadMemory();
    } else {
      reset();
      setMemory([]);
    }
  }, [conversationId, setCurrentConversation, setMessages, setIsLoading, reset]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMessage.trim() || isStreaming) return;

    const messageText = newMessage;
    setNewMessage('');

    // Add user message optimistically
    const tempUserMessage: Message = {
      id: `temp-${Date.now()}`,
      conversation_id: currentConversation?.id || '',
      role: 'user',
      content: messageText,
      sources: null,
      token_count: null,
      model_used: null,
      processing_time_ms: null,
      created_at: new Date().toISOString(),
    };
    addMessage(tempUserMessage);

    // Create streaming assistant message placeholder
    const streamingMessageId = `temp-stream-${Date.now()}`;
    const tempAssistantMessage: Message = {
      id: streamingMessageId,
      conversation_id: currentConversation?.id || '',
      role: 'assistant',
      content: '',
      sources: null,
      token_count: null,
      model_used: null,
      processing_time_ms: null,
      created_at: new Date().toISOString(),
    };
    addMessage(tempAssistantMessage);
    startStreamingMessage(streamingMessageId);

    try {
      await api.sendMessageStream(
        {
          message: messageText,
          conversation_id: currentConversation?.id,
          stream: true,
        },
        // onToken
        (token: string) => {
          appendStreamingToken(token);
        },
        // onComplete
        (data) => {
          // Update conversation if new
          if (!currentConversation && data.conversation_id) {
            navigate(`/chat?conversation=${data.conversation_id}`, { replace: true });
          }

          // Complete the streaming message
          const assistantMessage: Message = {
            id: `temp-${Date.now() + 1}`,
            conversation_id: data.conversation_id,
            role: 'assistant',
            content: streamingContent,
            sources: data.sources,
            token_count: data.token_count,
            model_used: 'qwen2.5:7b-instruct',
            processing_time_ms: data.processing_time_ms,
            created_at: new Date().toISOString(),
          };
          completeStreamingMessage(assistantMessage);

          // Refresh conversations list
          api.getConversations({ page_size: 50 }).then(response_convs => {
            setConversations(getPageItems(response_convs));
          }).catch(console.error);
        }
      );
    } catch (error) {
      console.error('Failed to send message:', error);
      cancelStreamingMessage();
    }
  };

  const handleNewConversation = () => {
    navigate('/chat', { replace: true });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setSelectedFile(file);
    setUploadError(null);
    setUploadSuccess(false);

    if (file) {
      // Validate file type
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      const allowed = ['.pdf', '.docx', '.txt', '.md'];
      if (!allowed.includes(ext)) {
        setUploadError('Unsupported file type. Supported: PDF, DOCX, TXT, MD');
        return;
      }

      // Validate file size (50MB max)
      const maxSize = 50 * 1024 * 1024;
      if (file.size > maxSize) {
        setUploadError(`File too large. Maximum size is 50MB`);
        return;
      }

      // Auto-upload
      handleUpload(file);
    }
  };

  const handleUpload = async (file: File) => {
    setUploading(true);
    setUploadError(null);
    setUploadSuccess(false);

    try {
      await api.uploadDocument(file, {
        title: file.name,
        description: 'Uploaded via Chat attachment',
        classification: 'public_internal',
      });
      setUploadSuccess(true);
      setSelectedFile(null);
      setTimeout(() => setUploadSuccess(false), 3000);
    } catch (error) {
      const err = error as { response?: { data?: { detail?: string } } };
      setUploadError(
        err?.response?.data?.detail || 'Upload failed. Please try again.'
      );
    } finally {
      setUploading(false);
    }
  };

  const handleAttachFile = () => {
    fileInputRef.current?.click();
  };

  const handleSelectConversation = (id: string) => {
    navigate(`/chat?conversation=${id}`);
  };

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      {/* Mobile sidebar toggle */}
      <button
        className="lg:hidden fixed bottom-4 right-4 z-50 p-3 rounded-full bg-primary-600 text-white shadow-lg"
        onClick={() => setShowSidebar(true)}
        aria-label="Open conversations"
      >
        <MessageSquare className="h-6 w-6" />
      </button>

      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversation?.id || null}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        loading={isLoading}
      />

      {/* Mobile sidebar overlay */}
      {showSidebar && (
        <>
          <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setShowSidebar(false)} />
          <aside className="fixed top-0 left-0 z-50 h-full w-72 bg-white border-r border-dark-200 lg:hidden animate-in">
            <div className="flex items-center justify-between p-4 border-b border-dark-200">
              <h2 className="font-semibold text-dark-900">Conversations</h2>
              <button onClick={() => setShowSidebar(false)} className="p-1 rounded hover:bg-dark-100">
                <ChevronRight className="h-5 w-5" />
              </button>
            </div>
            <Sidebar
              conversations={conversations}
              currentConversationId={currentConversation?.id || null}
              onSelectConversation={(id) => { handleSelectConversation(id); setShowSidebar(false); }}
              onNewConversation={() => { handleNewConversation(); setShowSidebar(false); }}
              loading={isLoading}
            />
          </aside>
        </>
      )}

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat header */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-dark-200 bg-white">
          <div className="flex items-center gap-3">
            <button
              className="lg:hidden p-2 rounded hover:bg-dark-100"
              onClick={() => setShowSidebar(true)}
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            <div>
              <h2 className="font-medium text-dark-900">
                {currentConversation?.title || 'New Conversation'}
              </h2>
              <p className="text-xs text-dark-500">
                {currentConversation
                  ? `${currentConversation.message_count} messages`
                  : 'Start a new conversation'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {memory.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                aria-label="Show conversation memory"
                onClick={() => setShowMemory(!showMemory)}
                className={clsx('gap-1', showMemory ? 'bg-primary-50 text-primary-700' : 'text-dark-500 hover:text-primary-700')}
              >
                <BookOpen className="h-4 w-4" />
                <span className="hidden sm:inline">Memory ({memory.length})</span>
              </Button>
            )}
            <Button variant="ghost" size="icon" aria-label="Settings">
              <Settings className="h-5 w-5" />
            </Button>
            <Button variant="ghost" size="icon" aria-label="More options">
              <MoreVertical className="h-5 w-5" />
            </Button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6" role="log" aria-live="polite">
          {messages.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center h-full text-center text-dark-500">
              <Bot className="h-16 w-16 text-dark-200 mb-4" />
              <h3 className="text-lg font-medium text-dark-700">Welcome to AegisAI</h3>
              <p className="mt-2 max-w-md">Ask me anything about your company documents. I'll search through authorized knowledge and provide answers with citations.</p>
              <div className="mt-6 flex flex-wrap gap-2 justify-center">
                <Button variant="secondary" size="sm" onClick={() => setNewMessage('What is the maintenance schedule for Machine X?')}>
                  Machine X maintenance
                </Button>
                <Button variant="secondary" size="sm" onClick={() => setNewMessage('Show me the safety protocols')}>
                  Safety protocols
                </Button>
                <Button variant="secondary" size="sm" onClick={() => setNewMessage('What are the HR policies?')}>
                  HR policies
                </Button>
              </div>
            </div>
          )}

          {messages.map((message) => {
            // For streaming message, show the streaming content
            const isStreamingMessage = message.id === streamingMessageId;
            const displayContent = isStreamingMessage ? streamingContent : message.content;

            return (
              <MessageBubble
                key={message.id}
                message={{ ...message, content: displayContent }}
                onCopy={(text) => navigator.clipboard.writeText(text)}
                onFeedback={() => {}}
              />
            );
          })}

          {isStreaming && streamingMessageId && !messages.some(m => m.id === streamingMessageId) && (
            <div className="flex gap-3">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
                <Bot className="h-5 w-5 text-primary-700" />
              </div>
              <div className="flex-1 max-w-[85%]">
                <div className="inline-block px-4 py-2 rounded-2xl bg-white border border-dark-200 rounded-bl-md">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Memory panel */}
        {showMemory && memory.length > 0 && (
          <div className="border-t border-dark-200 bg-dark-50 p-4 animate-in">
            <div className="max-w-4xl mx-auto">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium text-dark-900 flex items-center gap-2">
                  <BookOpen className="h-5 w-5 text-primary-600" />
                  Conversation Memory
                </h3>
                <button
                  onClick={() => setShowMemory(false)}
                  className="p-1 rounded hover:bg-dark-200 text-dark-400 hover:text-dark-600"
                  aria-label="Close memory panel"
                >
                  <ChevronRight className="h-5 w-5" />
                </button>
              </div>
              <p className="text-xs text-dark-500 mb-3">
                Summarized context from earlier conversation turns. This context is automatically included in new queries for better continuity.
              </p>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {memory.map((mem, idx) => (
                  <div
                    key={mem.id}
                    className="p-3 rounded-lg bg-white border border-dark-200"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-dark-700">
                        Summary {memory.length - idx} (oldest first)
                      </span>
                      <div className="flex items-center gap-2 text-xs text-dark-400">
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {mem.message_count} messages
                        </span>
                        <span className="flex items-center gap-1">
                          <span>~{mem.token_count} tokens</span>
                        </span>
                      </div>
                    </div>
                    <div className="text-sm text-dark-600 mb-2">{mem.summary}</div>
                    <div className="text-xs text-dark-400 flex items-center gap-3 flex-wrap">
                      <span>
                        {format(new Date(mem.window_start), 'MMM d, HH:mm')} -{' '}
                        {format(new Date(mem.window_end), 'MMM d, HH:mm')}
                      </span>
                      <span>Created: {format(new Date(mem.created_at), 'MMM d, HH:mm')}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Input area */}
        <div className="border-t border-dark-200 bg-white p-4">
          <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto">
            <div className="flex items-end gap-2">
              <input
                type="file"
                ref={fileInputRef}
                accept=".pdf,.docx,.txt,.md"
                onChange={handleFileChange}
                className="hidden"
                aria-label="Attach document"
              />
              <Button
                type="button"
                variant={uploadError ? 'danger' : 'ghost'}
                size="icon"
                aria-label="Attach file"
                onClick={handleAttachFile}
                disabled={uploading}
              >
                {uploading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Paperclip className="h-5 w-5" />}
              </Button>
              <Textarea
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                placeholder="Ask AegisAI..."
                className="flex-1 min-h-[44px] max-h-[200px] resize-none"
                rows={1}
              />
              <Button
                type="submit"
                variant="primary"
                size="icon"
                disabled={!newMessage.trim() || isStreaming}
                aria-label="Send message"
              >
                <Send className="h-5 w-5" />
              </Button>
            </div>
            {uploadSuccess && (
              <p className="text-xs text-green-600 text-center mt-2">
                Document uploaded successfully and is being processed
              </p>
            )}
            {uploadError && (
              <p className="text-xs text-red-600 text-center mt-2">
                {uploadError}
              </p>
            )}
            {selectedFile && !uploading && !uploadSuccess && (
              <p className="text-xs text-dark-500 text-center mt-2">
                Selected: {selectedFile.name}
              </p>
            )}
            <p className="text-xs text-dark-400 text-center mt-2">
              Press Enter to send, Shift+Enter for new line
            </p>
          </form>
        </div>
      </div>
    </div>
  );
};