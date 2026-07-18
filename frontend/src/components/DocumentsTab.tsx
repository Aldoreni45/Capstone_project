'use client';

import { useState } from 'react';
import { Upload, CheckCircle, AlertCircle, FileText } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { IngestionResponse } from '@/types';

export default function DocumentsTab() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<IngestionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.type !== 'application/pdf') {
        setError('Only PDF files are supported');
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setError(null);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiClient.ingestPdf(file);
      setResult(response);
      setFile(null);
    } catch (err) {
      setError('Failed to ingest PDF. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-2xl">
      <h2 className="text-xl font-semibold mb-6">Document Ingest & Corpus</h2>

      {/* Upload Area */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-6 mb-6">
        <div className="border-2 border-dashed border-dark-border rounded-lg p-8 text-center">
          <Upload className="w-12 h-12 mx-auto mb-4 text-dark-text" />
          <p className="text-dark-text mb-4">
            Upload a PDF document to ingest into the vector database
          </p>
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            className="hidden"
            id="file-upload"
            disabled={uploading}
          />
          <label
            htmlFor="file-upload"
            className="inline-block px-6 py-3 bg-primary text-dark-bg rounded-lg font-semibold cursor-pointer hover:bg-primary-dark transition-colors"
          >
            Select PDF File
          </label>
          {file && (
            <p className="mt-4 text-sm text-primary">
              Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          )}
        </div>

        {file && (
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="w-full mt-4 px-6 py-3 bg-primary text-dark-bg rounded-lg font-semibold hover:bg-primary-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {uploading ? 'Ingesting...' : 'Ingest Document'}
          </button>
        )}
      </div>

      {/* Result */}
      {result && (
        <div className="bg-dark-card border border-dark-border rounded-lg p-6 mb-6">
          <div className="flex items-start gap-3">
            <CheckCircle className="w-6 h-6 text-green-400 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-lg mb-2">Ingestion Successful</h3>
              <div className="space-y-1 text-sm text-dark-text">
                <p><strong>Document ID:</strong> {result.document_id}</p>
                <p><strong>Filename:</strong> {result.filename}</p>
                <p><strong>Chunks Created:</strong> {result.chunks_created}</p>
                <p><strong>Status:</strong> {result.status}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-lg mb-2 text-red-400">Error</h3>
              <p className="text-sm">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Info */}
      <div className="bg-dark-card border border-dark-border rounded-lg p-6">
        <div className="flex items-start gap-3">
          <FileText className="w-6 h-6 text-primary flex-shrink-0 mt-1" />
          <div>
            <h3 className="font-semibold text-lg mb-2">About Document Ingestion</h3>
            <p className="text-sm text-dark-text mb-2">
              Uploaded PDFs are processed through the following pipeline:
            </p>
            <ol className="text-sm text-dark-text list-decimal list-inside space-y-1">
              <li>PDF loading with metadata extraction</li>
              <li>Document chunking (Recursive strategy)</li>
              <li>Embedding generation (HuggingFace)</li>
              <li>Vector storage in Weaviate database</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
