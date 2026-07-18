'use client';

import { X, Cpu, Database, Brain, MessageSquare } from 'lucide-react';

interface SettingsSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsSidebar({ isOpen, onClose }: SettingsSidebarProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50"
        onClick={onClose}
      />

      {/* Sidebar */}
      <div className="relative w-80 bg-dark-card border-r border-dark-border h-full overflow-y-auto">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-primary">Settings Engine</h2>
            <button
              onClick={onClose}
              className="p-2 hover:bg-dark-border rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Pipeline Settings */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-dark-text mb-3 flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              Pipeline Architectures
            </h3>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-dark-text mb-1">Embedding Model</label>
                <select className="w-full bg-dark-bg border border-dark-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary">
                  <option value="huggingface">HuggingFace (bge-large-en-v1.5)</option>
                  <option value="bge">BGE Small</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-dark-text mb-1">Retrieval Strategy</label>
                <select className="w-full bg-dark-bg border border-dark-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary">
                  <option value="hybrid">Hybrid Search</option>
                  <option value="cosine">Cosine Similarity</option>
                  <option value="mmr">MMR</option>
                  <option value="multiquery">Multi-Query</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-dark-text mb-1">LLM Model</label>
                <select className="w-full bg-dark-bg border border-dark-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary">
                  <option value="llama-3-3-70b">Llama 3.3 70B</option>
                  <option value="deepseek-r1-70b">DeepSeek R1 70B</option>
                  <option value="gemma-2-9b">Gemma 2 9B</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-dark-text mb-1">Conversational Memory</label>
                <select className="w-full bg-dark-bg border border-dark-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary">
                  <option value="buffer">Buffer Memory</option>
                  <option value="summary">Summary Memory</option>
                  <option value="token">Token Memory</option>
                </select>
              </div>
            </div>
          </div>

          {/* Chunking Settings */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-dark-text mb-3 flex items-center gap-2">
              <Database className="w-4 h-4" />
              Chunking Strategies
            </h3>
            <div>
              <label className="block text-xs text-dark-text mb-1">Active Chunker</label>
              <select className="w-full bg-dark-bg border border-dark-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary">
                <option value="recursive">Recursive</option>
                <option value="semantic">Semantic</option>
              </select>
            </div>
          </div>

          {/* Diagnostics */}
          <div>
            <h3 className="text-sm font-semibold text-dark-text mb-3 flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Diagnostics
            </h3>
            <button className="w-full px-4 py-2 bg-primary text-dark-bg rounded-lg font-semibold hover:bg-primary-dark transition-colors text-sm">
              Reset Chat & Memory
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
