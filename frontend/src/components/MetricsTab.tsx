'use client';

import { Activity, TrendingUp, Clock, Database } from 'lucide-react';

export default function MetricsTab() {
  return (
    <div className="max-w-4xl">
      <h2 className="text-xl font-semibold mb-6">Performance Metrics</h2>

      {/* Metrics Grid */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-dark-card border border-dark-border rounded-lg p-4 text-center">
          <Activity className="w-8 h-8 mx-auto mb-2 text-primary" />
          <div className="text-2xl font-bold text-primary">87%</div>
          <div className="text-xs text-dark-text">Faithfulness</div>
        </div>
        <div className="bg-dark-card border border-dark-border rounded-lg p-4 text-center">
          <TrendingUp className="w-8 h-8 mx-auto mb-2 text-primary" />
          <div className="text-2xl font-bold text-primary">60%</div>
          <div className="text-xs text-dark-text">Context Precision</div>
        </div>
        <div className="bg-dark-card border border-dark-border rounded-lg p-4 text-center">
          <Clock className="w-8 h-8 mx-auto mb-2 text-primary" />
          <div className="text-2xl font-bold text-primary">33%</div>
          <div className="text-xs text-dark-text">Context Recall</div>
        </div>
        <div className="bg-dark-card border border-dark-border rounded-lg p-4 text-center">
          <Database className="w-8 h-8 mx-auto mb-2 text-primary" />
          <div className="text-2xl font-bold text-primary">75%</div>
          <div className="text-xs text-dark-text">Answer Relevancy</div>
        </div>
      </div>

      {/* Additional Metrics */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">RAGAS & DeepEval Quality Metrics Dashboard</h3>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-dark-text">Semantic Similarity</span>
              <span className="text-primary">85.8%</span>
            </div>
            <div className="h-2 bg-dark-border rounded-full overflow-hidden">
              <div className="h-full bg-primary" style={{ width: '85.8%' }} />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-dark-text">Hallucination Score</span>
              <span className="text-red-400">38.7%</span>
            </div>
            <div className="h-2 bg-dark-border rounded-full overflow-hidden">
              <div className="h-full bg-red-400" style={{ width: '38.7%' }} />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-dark-text">Retrieval Accuracy</span>
              <span className="text-yellow-400">25.0%</span>
            </div>
            <div className="h-2 bg-dark-border rounded-full overflow-hidden">
              <div className="h-full bg-yellow-400" style={{ width: '25%' }} />
            </div>
          </div>
        </div>
      </div>

      {/* System Health */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">System Health</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
            <div className="text-sm text-dark-text mb-1">Pipeline Status</div>
            <div className="text-lg font-semibold text-green-400">Healthy</div>
          </div>
          <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
            <div className="text-sm text-dark-text mb-1">Vector DB</div>
            <div className="text-lg font-semibold text-green-400">Connected</div>
          </div>
          <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
            <div className="text-sm text-dark-text mb-1">Documents Indexed</div>
            <div className="text-lg font-semibold text-primary">2</div>
          </div>
          <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
            <div className="text-sm text-dark-text mb-1">Last Query</div>
            <div className="text-lg font-semibold text-primary">2s ago</div>
          </div>
        </div>
      </div>
    </div>
  );
}
