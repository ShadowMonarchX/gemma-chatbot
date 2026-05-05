import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import App from '../App';
import { useChatStore } from '../stores/chatStore';

const encoder = new TextEncoder();

const modelsPayload = {
  active_model_id: 'gemma-2b',
  models: [
    {
      id: 'gemma-2b',
      label: 'Gemma 2B',
      backend: 'mlx',
      source: 'google/gemma-2b-it',
      quantization: 'INT4',
      available: true,
      default: true,
      description: 'Fast baseline model.',
      alias_of: null,
    },
    {
      id: 'gemma-e2b',
      label: 'Gemma E2B',
      backend: 'mlx',
      source: 'google/gemma-2b-it',
      quantization: 'INT4',
      available: true,
      default: false,
      description: 'Efficient profile.',
      alias_of: 'gemma-2b',
    },
    {
      id: 'gemma-e4b',
      label: 'Gemma E4B',
      backend: 'mlx',
      source: 'google/gemma-2b-it',
      quantization: 'INT4',
      available: true,
      default: false,
      description: 'Quality profile.',
      alias_of: 'gemma-2b',
    },
  ],
};

const createSseResponse = (events: string[], responseMs = 312): Response => {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      events.forEach((event) => {
        controller.enqueue(encoder.encode(`${event}\n\n`));
      });
      controller.close();
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'X-Response-Ms': String(responseMs),
      'X-Request-Id': 'req-chat-1',
    },
  });
};

const createJsonResponse = (payload: unknown, status = 200, requestId = 'req-json-1'): Response =>
  new Response(JSON.stringify(payload), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'X-Request-Id': requestId,
    },
  });

const installDefaultFetch = (chatResponse: Response): void => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/models')) {
        return Promise.resolve(createJsonResponse(modelsPayload));
      }
      if (url.includes('/api/chat')) {
        return Promise.resolve(chatResponse);
      }
      return Promise.resolve(createJsonResponse({ error: 'not found' }, 404));
    })
  );
};

const resetStore = () => {
  const actions = useChatStore.getState().actions;
  useChatStore.setState({
    messages: [],
    skill: { id: 'chat', label: 'Chat' },
    modelId: 'gemma-2b',
    models: [],
    modelsLoading: false,
    isStreaming: false,
    responseMs: null,
    inputError: null,
    toasts: [],
    actions,
  });
};

describe('Chat page', () => {
  beforeEach(() => {
    resetStore();
    vi.restoreAllMocks();
  });

  const renderChatApp = () =>
    render(
      <MemoryRouter
        initialEntries={['/chat']}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <App />
      </MemoryRouter>
    );

  it('renders skill tabs', async () => {
    installDefaultFetch(createSseResponse(['data: [DONE]']));

    renderChatApp();

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /switch to chat mode/i })).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: /switch to code mode/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/select model/i)).toBeInTheDocument();
  });

  it('sends message on Enter', async () => {
    installDefaultFetch(createSseResponse(['data: Hello', 'data: [DONE]']));

    renderChatApp();

    const textarea = await screen.findByLabelText(/message input/i);
    fireEvent.change(textarea, { target: { value: 'Hello model' } });
    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('Hello model')).toBeInTheDocument();
    });
  });

  it('expands message input for large content', async () => {
    installDefaultFetch(createSseResponse(['data: [DONE]']));

    renderChatApp();

    const textarea = await screen.findByLabelText(/message input/i);
    Object.defineProperty(textarea, 'scrollHeight', {
      configurable: true,
      value: 320,
    });

    fireEvent.change(textarea, {
      target: { value: ['First line', 'Second line', 'Third line'].join('\n') },
    });

    await waitFor(() => {
      expect(textarea).toHaveStyle({ height: '240px', overflowY: 'auto' });
    });
  });

  it('shows typing indicator while streaming', async () => {
    let resolveChat: (value: Response | PromiseLike<Response>) => void = () => {};

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/api/models')) {
          return Promise.resolve(createJsonResponse(modelsPayload));
        }
        if (url.includes('/api/chat')) {
          return new Promise<Response>((resolve) => {
            resolveChat = resolve;
          });
        }
        return Promise.resolve(createJsonResponse({ error: 'not found' }, 404));
      })
    );

    renderChatApp();

    const textarea = await screen.findByLabelText(/message input/i);
    fireEvent.change(textarea, { target: { value: 'Stream please' } });
    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText(/assistant is typing/i)).toBeInTheDocument();
    });

    resolveChat(createSseResponse(['data: hello', 'data: [DONE]']));

    await waitFor(() => {
      expect(screen.queryByText(/assistant is typing/i)).not.toBeInTheDocument();
    });
  });

  it('displays response time badge after stream ends', async () => {
    installDefaultFetch(createSseResponse(['data: Hello', 'data: world', 'data: [DONE]'], 312));

    renderChatApp();

    const textarea = await screen.findByLabelText(/message input/i);
    fireEvent.change(textarea, { target: { value: 'Latency check' } });
    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText(/312 ms/i)).toBeInTheDocument();
    });
  });

  it('clears history on skill switch', async () => {
    installDefaultFetch(createSseResponse(['data: done', 'data: [DONE]']));

    renderChatApp();

    const textarea = await screen.findByLabelText(/message input/i);
    fireEvent.change(textarea, { target: { value: 'Keep this' } });
    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('Keep this')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: /switch to code mode/i }));

    await waitFor(() => {
      expect(screen.queryByText('Keep this')).not.toBeInTheDocument();
    });
  });

  it('blocks empty message send', async () => {
    const fetchSpy = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/models')) {
        return Promise.resolve(createJsonResponse(modelsPayload));
      }
      if (url.includes('/api/chat')) {
        return Promise.resolve(createSseResponse(['data: [DONE]']));
      }
      return Promise.resolve(createJsonResponse({ error: 'not found' }, 404));
    });
    vi.stubGlobal('fetch', fetchSpy);

    renderChatApp();

    await screen.findByLabelText(/message input/i);
    expect(screen.getByRole('button', { name: /send message/i })).toBeDisabled();
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it('shows error toast on ApiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/api/models')) {
          return Promise.resolve(createJsonResponse(modelsPayload));
        }
        if (url.includes('/api/chat')) {
          return Promise.resolve(
            createJsonResponse({ error: 'Server failure' }, 500, 'req-error-1')
          );
        }
        return Promise.resolve(createJsonResponse({ error: 'not found' }, 404));
      })
    );

    renderChatApp();

    const textarea = await screen.findByLabelText(/message input/i);
    fireEvent.change(textarea, { target: { value: 'Trigger error' } });
    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText(/server failure/i)).toBeInTheDocument();
    });
  });

  it('shows error toast on SSE error event', async () => {
    installDefaultFetch(createSseResponse(['event: error\ndata: Token generation failed']));

    renderChatApp();

    const textarea = await screen.findByLabelText(/message input/i);
    fireEvent.change(textarea, { target: { value: 'Trigger stream error' } });
    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText(/token generation failed/i)).toBeInTheDocument();
    });
  });
});
