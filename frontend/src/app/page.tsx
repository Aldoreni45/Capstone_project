'use client';

import { useState } from 'react';
import { MessageSquare, FileText, BarChart3, Target, Settings } from 'lucide-react';
import ChatTab from '@/components/ChatTab';
import DocumentsTab from '@/components/DocumentsTab';
import MetricsTab from '@/components/MetricsTab';
import EvaluationTab from '@/components/EvaluationTab';
import SettingsSidebar from '@/components/SettingsSidebar';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'chat' | 'docs' | 'metrics' | 'eval'>('chat');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-dark-bg text-white">
      {/* Header */}
      <header className="border-b border-dark-border bg-dark-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 hover:bg-dark-border rounded-lg transition-colors"
              >
                <Settings className="w-6 h-6 text-primary" />
              </button>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-primary-dark bg-clip-text text-transparent">
                  Research Paper Answer Bot
                </h1>
                <p className="text-dark-text text-sm">
                  Production-Grade Retrieval-Augmented Generation (RAG) Architecture
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <SettingsSidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

        {/* Main Content */}
        <main className="flex-1 p-6">
          {/* Tabs */}
          <div className="flex gap-2 mb-6 border-b border-dark-border pb-4">
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeTab === 'chat'
                  ? 'bg-primary text-dark-bg font-semibold'
                  : 'bg-dark-card text-dark-text hover:bg-dark-border'
              }`}
            >
              <MessageSquare className="w-5 h-5" />
              Chat Assistant
            </button>
            <button
              onClick={() => setActiveTab('docs')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeTab === 'docs'
                  ? 'bg-primary text-dark-bg font-semibold'
                  : 'bg-dark-card text-dark-text hover:bg-dark-border'
              }`}
            >
              <FileText className="w-5 h-5" />
              Document Ingest
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeTab === 'metrics'
                  ? 'bg-primary text-dark-bg font-semibold'
                  : 'bg-dark-card text-dark-text hover:bg-dark-border'
              }`}
            >
              <BarChart3 className="w-5 h-5" />
              Performance Metrics
            </button>
            <button
              onClick={() => setActiveTab('eval')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeTab === 'eval'
                  ? 'bg-primary text-dark-bg font-semibold'
                  : 'bg-dark-card text-dark-text hover:bg-dark-border'
              }`}
            >
              <Target className="w-5 h-5" />
              RAG Evaluations
            </button>
          </div>

          {/* Tab Content */}
          {activeTab === 'chat' && <ChatTab />}
          {activeTab === 'docs' && <DocumentsTab />}
          {activeTab === 'metrics' && <MetricsTab />}
          {activeTab === 'eval' && <EvaluationTab />}
        </main>
      </div>
    </div>
  );
}
