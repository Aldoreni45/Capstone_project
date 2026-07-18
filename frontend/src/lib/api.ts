import axios from 'axios';
import { QueryRequest, QueryResponse, IngestionResponse, HealthResponse } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiClient = {
  // Health check
  async healthCheck(): Promise<HealthResponse> {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  },

  // Query endpoint
  async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await api.post<QueryResponse>('/query', request);
    return response.data;
  },

  // Ingest PDF endpoint
  async ingestPdf(file: File): Promise<IngestionResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post<IngestionResponse>('/ingest', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

export default apiClient;
