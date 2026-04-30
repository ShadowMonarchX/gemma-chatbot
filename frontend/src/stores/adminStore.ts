import { create } from 'zustand';

import { ApiError, apiCall } from '../api/client';
import { AdminResponseSchema, type AdminResponse } from '../api/types';

interface AdminActions {
  fetchAdmin: () => Promise<void>;
}

interface AdminState {
  data: AdminResponse | null;
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
  actions: AdminActions;
}

export const useAdminStore = create<AdminState>()((set) => ({
  data: null,
  loading: false,
  error: null,
  lastUpdated: null,
  actions: {
    fetchAdmin: async () => {
      set({ loading: true, error: null });
      try {
        const payload = await apiCall<AdminResponse>('/admin', { method: 'GET' });
        const parsed = AdminResponseSchema.parse(payload);
        set({
          data: parsed,
          loading: false,
          error: null,
          lastUpdated: new Date().toISOString(),
        });
      } catch (error) {
        const normalized =
          error instanceof ApiError
            ? error
            : new ApiError(500, 'unknown-request-id', (error as Error).message || 'Unknown error');
        set({
          loading: false,
          error: `${normalized.message} (${normalized.requestId})`,
        });
      }
    },
  },
}));
