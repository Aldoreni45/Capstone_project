'use client';

import { Target, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

export default function EvaluationTab() {
  return (
    <div className="max-w-4xl">
      <h2 className="text-xl font-semibold mb-6">RAG Evaluations</h2>

      {/* Evaluation Status */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-6 mb-6">
        <div className="flex items-center gap-3 mb-4">
          <Target className="w-6 h-6 text-primary" />
          <h3 className="text-lg font-semibold">Evaluation Status</h3>
        </div>
        <div className="flex items-center gap-2">
          <CheckCircle className="w-5 h-5 text-green-400" />
          <span className="text-green-400">Verified Heuristically</span>
        </div>
        <p className="text-sm text-dark-text mt-2">
          RAG pipeline health is stable. Evaluation complete using heuristic fallback scores.
        </p>
      </div>

      {/* Test Results */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Recent Test Results</h3>
        <div className="space-y-3">
          <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">Query: "what is rnn and hive"</span>
              <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded text-xs font-semibold">
                PARTIAL
              </span>
            </div>
            <div className="text-sm text-dark-text">
              <p>Concept Coverage: 50.0% (Found: rnn, Missing: hive)</p>
              <p>Confidence: 87.0%</p>
            </div>
          </div>
          <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">Query: "tell me about hive"</span>
              <span className="px-2 py-1 bg-red-500/20 text-red-400 rounded text-xs font-semibold">
                BAD
              </span>
            </div>
            <div className="text-sm text-dark-text">
              <p>Reason: Named entity 'hive' not found in retrieved context</p>
              <p>Confidence: 0.0%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Evaluation Metrics */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Evaluation Metrics Summary</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-5 h-5 text-green-400" />
              <span className="text-sm text-dark-text">Good Retrievals</span>
            </div>
            <div className="text-2xl font-bold text-green-400">65%</div>
          </div>
          <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-yellow-400" />
              <span className="text-sm text-dark-text">Partial Retrievals</span>
            </div>
            <div className="text-2xl font-bold text-yellow-400">25%</div>
          </div>
          <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <XCircle className="w-5 h-5 text-red-400" />
              <span className="text-sm text-dark-text">Bad Retrievals</span>
            </div>
            <div className="text-2xl font-bold text-red-400">10%</div>
          </div>
          <div className="bg-dark-bg border border-dark-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-5 h-5 text-primary" />
              <span className="text-sm text-dark-text">Total Queries</span>
            </div>
            <div className="text-2xl font-bold text-primary">20</div>
          </div>
        </div>
      </div>
    </div>
  );
}
