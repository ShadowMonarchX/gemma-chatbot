import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import Admin from '../pages/Admin';
import { useAdminStore } from '../stores/adminStore';

const adminPayload = {
  status: 'ok',
  model: 'google/gemma-4-2b-it',
  quantization: 'INT4',
  hardware: {
    chip: 'Apple M2',
    ram_total_gb: 16,
    ram_available_gb: 9.2,
    cpu_cores: 8,
    metal_gpu: true,
  },
  model_load_ms: 1200,
  avg_tokens_per_sec: 42.3,
  uptime_seconds: 3720,
  last_request_ms: 318,
  total_requests: 148,
  errors: 0,
  avg_response_ms: 318,
  requests_per_minute: 10,
  skill_usage: {
    chat: 73,
    code: 27,
  },
  hallucination_guards_triggered: 0,
  rate_limit_hits: 0,
};

const resetStore = () => {
  const actions = useAdminStore.getState().actions;
  useAdminStore.setState({
    data: null,
    loading: false,
    error: null,
    lastUpdated: null,
    actions,
  });
};

describe('Admin page', () => {
  beforeEach(() => {
    resetStore();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('renders hardware cards', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(adminPayload), {
          status: 200,
          headers: { 'Content-Type': 'application/json', 'X-Request-Id': 'req-admin-1' },
        })
      )
    );

    render(<Admin />);

    await waitFor(() => {
      expect(screen.getByText(/Apple M2/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Metal GPU/i)).toBeInTheDocument();
    expect(screen.getByText(/INT4/i)).toBeInTheDocument();
  });

  it('shows health status badge', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(adminPayload), {
          status: 200,
          headers: { 'Content-Type': 'application/json', 'X-Request-Id': 'req-admin-2' },
        })
      )
    );

    render(<Admin />);

    await waitFor(() => {
      expect(screen.getByText(/All systems nominal/i)).toBeInTheDocument();
    });
  });

  it('auto-refreshes every 10 seconds', async () => {
    vi.useFakeTimers();
    const fetchSpy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(adminPayload), {
        status: 200,
        headers: { 'Content-Type': 'application/json', 'X-Request-Id': 'req-admin-3' },
      })
    );
    vi.stubGlobal('fetch', fetchSpy);

    render(<Admin />);

    await Promise.resolve();
    await Promise.resolve();
    expect(fetchSpy).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(10000);
    await Promise.resolve();
    await Promise.resolve();
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  it('refresh button re-fetches data', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(adminPayload), {
        status: 200,
        headers: { 'Content-Type': 'application/json', 'X-Request-Id': 'req-admin-4' },
      })
    );
    vi.stubGlobal('fetch', fetchSpy);

    render(<Admin />);

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledTimes(1);
    });

    fireEvent.click(screen.getByRole('button', { name: /refresh admin metrics/i }));

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledTimes(2);
    });
  });
});
