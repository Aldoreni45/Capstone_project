'use client';

import { useState } from 'react';
import { Send, BookOpen, Globe, RefreshCw } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { ChatMessage, QueryResponse } from '@/types';

export default function ChatTab() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [lastResponse, setLastResponse] = useState<QueryResponse | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await apiClient.query({
        query: input,
        mode: 'rag',
        memory_type: 'buffer',
      });

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.answer,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setLastResponse(response);
    } catch (error) {
      console.error('Error querying:', error);
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your query. Please try again.',
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setMessages([]);
    setLastResponse(null);
  };

  return (
    <div className="flex gap-6">
      {/* Chat Messages */}
      <div className="flex-1">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Chat History</h2>
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-3 py-1.5 bg-dark-card border border-dark-border rounded-lg hover:bg-dark-border transition-colors text-sm"
          >
            <RefreshCw className="w-4 h-4" />
            Reset Chat
          </button>
        </div>

        <div className="bg-dark-card border border-dark-border rounded-lg p-4 h-[500px] overflow-y-auto mb-4">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-dark-text">
              <p>No messages yet. Start a conversation!</p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${
                    msg.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-3 ${
                      msg.role === 'user'
                        ? 'bg-primary text-dark-bg'
                        : 'bg-dark-border text-white'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-dark-border rounded-lg p-3">
                    <p className="text-dark-text">Processing your query...</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input */}
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the papers..."
            className="flex-1 bg-dark-card border border-dark-border rounded-lg px-4 py-3 focus:outline-none focus:border-primary transition-colors"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-6 py-3 bg-primary text-dark-bg rounded-lg font-semibold hover:bg-primary-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>

      {/* Context & Citations */}
      <div className="w-96">
        <h2 className="text-xl font-semibold mb-4">Context & Citations</h2>

        {lastResponse ? (
          <div className="space-y-4">
            {/* Confidence Score */}
            <div className="bg-dark-card border border-dark-border rounded-lg p-4">
              <h3 className="text-sm text-dark-text mb-2">Confidence Score</h3>
              <div className="flex items-center gap-3">
                <div className="text-3xl font-bold text-primary">
                  {(lastResponse.confidence * 100).toFixed(1)}%
                </div>
                <div className="flex-1 h-2 bg-dark-border rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all"
                    style={{ width: `${lastResponse.confidence * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Retrieval Quality */}
            <div className="bg-dark-card border border-dark-border rounded-lg p-4">
              <h3 className="text-sm text-dark-text mb-2">Retrieval Quality</h3>
              <div className="flex items-center gap-2">
                <span
                  className={`px-3 py-1 rounded-full text-sm font-semibold ${
                    lastResponse.retrieval_quality === 'GOOD'
                      ? 'bg-green-500/20 text-green-400'
                      : lastResponse.retrieval_quality === 'PARTIAL'
                      ? 'bg-yellow-500/20 text-yellow-400'
                      : 'bg-red-500/20 text-red-400'
                  }`}
                >
                  {lastResponse.retrieval_quality}
                </span>
                {lastResponse.query_rewritten && (
                  <span className="flex items-center gap-1 text-xs text-dark-text">
                    <RefreshCw className="w-3 h-3" />
                    Query Rewritten
                  </span>
                )}
                {lastResponse.web_search_used && (
                  <span className="flex items-center gap-1 text-xs text-dark-text">
                    <Globe className="w-3 h-3" />
                    Web Search
                  </span>
                )}
              </div>
              <p className="text-xs text-dark-text mt-2">{lastResponse.retrieval_reason}</p>
            </div>

            {/* Citations */}
            {lastResponse.citations.length > 0 ? (
              <div className="bg-dark-card border border-dark-border rounded-lg p-4">
                <h3 className="text-sm text-dark-text mb-3 flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
                  Top Cited Sources
                </h3>
                <div className="space-y-2">
                  {lastResponse.citations.slice(0, 3).map((citation, idx) => (
                    <details
                      key={idx}
                      className="bg-dark-bg border border-dark-border rounded-lg p-3"
                    >
                      <summary className="cursor-pointer font-medium text-sm">
                        Reference {idx + 1}: {citation.title} - Page {citation.page}
                      </summary>
                      <div className="mt-2 text-xs text-dark-text space-y-1">
                        <p>
                          <strong>Chunk ID:</strong> {citation.chunk_id}
                        </p>
                        <p>
                          <strong>Source:</strong> {citation.source}
                        </p>
                        <p>
                          <strong>Passage:</strong>
                        </p>
                        <p className="italic">"{citation.passage}"</p>
                      </div>
                    </details>
                  ))}
                </div>
              </div>
            ) : (
              <div className="bg-dark-card border border-dark-border rounded-lg p-4">
                <p className="text-dark-text text-sm">
                  No citations available (retrieval quality was BAD)
                </p>
              </div>
            )}
          </div>
        ) : (
          <div className="bg-dark-card border border-dark-border rounded-lg p-4">
            <p className="text-dark-text text-sm">
              Submit a query to see context and citations.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
