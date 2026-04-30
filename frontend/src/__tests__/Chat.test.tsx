import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import App from '../App';
import { useChatStore } from '../stores/chatStore';

const encoder = new TextEncoder();

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
      'X-Request-Id': 'req-test-1',
    },
  });
};

const resetStore = () => {
  const actions = useChatStore.getState().actions;
  useChatStore.setState({
    messages: [],
    skill: { id: 'chat', label: 'Chat' },
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
      <MemoryRouter initialEntries={['/chat']}>
        <App />
      </MemoryRouter>
    );

  it('renders skill tabs', () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(createSseResponse(['data: [DONE]'])));
    renderChatApp();

    expect(screen.getByRole('button', { name: /switch to chat mode/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /switch to code mode/i })).toBeInTheDocument();
  });

  it('sends message on Enter', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(createSseResponse(['data: Hello', 'data: [DONE]']))
    );

    renderChatApp();

    const textarea = screen.getByLabelText(/message input/i);
    fireEvent.change(textarea, { target: { value: 'Hello model' } });
    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('Hello model')).toBeInTheDocument();
    });
  });

  it('shows typing indicator while streaming', async () => {
    let resolveFetch: (value: Response | PromiseLike<Response>) => void = () => {};
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        () =>
          new Promise<Response>((resolve) => {
            resolveFetch = resolve;
          })
      )
    );

    renderChatApp();

    const textarea = screen.getByLabelText(/message input/i);
    fireEvent.change(textarea, { target: { value: 'Stream please' } });
    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText(/assistant is typing/i)).toBeInTheDocument();
    });

    resolveFetch(createSseResponse(['data: hello', 'data: [DONE]']));

    await waitFor(() => {
      expect(screen.queryByText(/assistant is typing/i)).not.toBeInTheDocument();
    });
  });

  it('displays response time badge after stream ends', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        createSseResponse(['data: Hello', 'data: world', 'data: [DONE]'], 312)
      )
    );

    renderChatApp();

    const textarea = screen.getByLabelText(/message input/i);
    fireEvent.change(textarea, { target: { value: 'Latency check' } });
    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText(/312 ms/i)).toBeInTheDocument();
    });
  });

  it('clears history on skill switch', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(createSseResponse(['data: done', 'data: [DONE]'])));

    renderChatApp();

    const textarea = screen.getByLabelText(/message input/i);
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
    const fetchSpy = vi.fn().mockResolvedValue(createSseResponse(['data: [DONE]']));
    vi.stubGlobal('fetch', fetchSpy);

    renderChatApp();

    expect(screen.getByRole('button', { name: /send message/i })).toBeDisabled();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('shows error toast on ApiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: 'Server failure' }), {
          status: 500,
          headers: {
            'Content-Type': 'application/json',
            'X-Request-Id': 'req-error-1',
          },
        })
      )
    );

    renderChatApp();

    const textarea = screen.getByLabelText(/message input/i);
    fireEvent.change(textarea, { target: { value: 'Trigger error' } });
    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText(/server failure/i)).toBeInTheDocument();
    });
  });
});
