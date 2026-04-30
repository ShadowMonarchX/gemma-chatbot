import { create } from 'zustand';

import { ApiError, apiStream } from '../api/client';
import {
  ChatRequestSchema,
  type ChatRequest,
  type SkillId,
  type UiMessage,
} from '../api/types';

export interface ToastItem {
  id: string;
  kind: 'error' | 'success';
  message: string;
}

export interface SkillOption {
  id: SkillId;
  label: string;
}

interface ChatActions {
  sendMessage: (content: string) => Promise<void>;
  setSkill: (skill: SkillOption) => void;
  clearHistory: () => void;
  appendToken: (token: string) => void;
  setResponseMs: (ms: number | null) => void;
  pushToast: (kind: 'error' | 'success', message: string) => void;
  dismissToast: (id: string) => void;
}

interface ChatState {
  messages: UiMessage[];
  skill: SkillOption;
  isStreaming: boolean;
  responseMs: number | null;
  inputError: string | null;
  toasts: ToastItem[];
  actions: ChatActions;
}

const DEFAULT_SKILL: SkillOption = { id: 'chat', label: 'Chat' };
const MAX_MESSAGE_LENGTH = 4096;

export const useChatStore = create<ChatState>()((set, get) => ({
  messages: [],
  skill: DEFAULT_SKILL,
  isStreaming: false,
  responseMs: null,
  inputError: null,
  toasts: [],
  actions: {
    setSkill: (skill: SkillOption) => {
      set({
        skill,
        messages: [],
        isStreaming: false,
        responseMs: null,
        inputError: null,
      });
    },
    clearHistory: () => {
      set({
        messages: [],
        responseMs: null,
        inputError: null,
      });
    },
    appendToken: (token: string) => {
      set((state) => {
        if (state.messages.length === 0) {
          return state;
        }
        const updated = [...state.messages];
        const last = updated[updated.length - 1];
        if (last.role !== 'assistant') {
          return state;
        }
        updated[updated.length - 1] = {
          ...last,
          content: `${last.content}${token}`,
        };
        return { messages: updated };
      });
    },
    setResponseMs: (ms: number | null) => {
      set((state) => {
        if (state.messages.length === 0) {
          return { responseMs: ms };
        }
        const updated = [...state.messages];
        const last = updated[updated.length - 1];
        if (last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            responseMs: ms,
          };
        }
        return { responseMs: ms, messages: updated };
      });
    },
    pushToast: (kind: 'error' | 'success', message: string) => {
      const toast: ToastItem = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        kind,
        message,
      };
      set((state) => ({ toasts: [...state.toasts, toast] }));
    },
    dismissToast: (id: string) => {
      set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) }));
    },
    sendMessage: async (content: string) => {
      const state = get();
      if (state.isStreaming) {
        return;
      }

      const trimmed = content.trim();
      if (!trimmed) {
        set({ inputError: 'Message cannot be empty.' });
        get().actions.pushToast('error', 'Message cannot be empty.');
        return;
      }
      if (trimmed.length > MAX_MESSAGE_LENGTH) {
        set({ inputError: `Message must be <= ${MAX_MESSAGE_LENGTH} characters.` });
        get().actions.pushToast('error', 'Message is too long.');
        return;
      }

      const userMessage: UiMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: trimmed,
      };
      const assistantMessage: UiMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '',
        skillId: state.skill.id,
        responseMs: null,
      };

      const nextMessages = [...state.messages, userMessage, assistantMessage];
      const requestPayload: ChatRequest = {
        messages: nextMessages
          .filter((message) => message.role === 'user' || message.role === 'assistant')
          .map((message) => ({ role: message.role, content: message.content }))
          .filter((message) => message.content.length > 0)
          .slice(-20),
        skill_id: state.skill.id,
        stream: true,
      };

      const parseResult = ChatRequestSchema.safeParse(requestPayload);
      if (!parseResult.success) {
        const issue = parseResult.error.issues[0];
        const message = issue ? issue.message : 'Invalid request payload.';
        set({ inputError: message });
        get().actions.pushToast('error', message);
        return;
      }

      set({
        messages: nextMessages,
        isStreaming: true,
        inputError: null,
        responseMs: null,
      });

      try {
        const response = await apiStream('/chat', {
          method: 'POST',
          body: JSON.stringify(parseResult.data),
        });

        const headerMs = Number(response.headers.get('X-Response-Ms') ?? '0');
        const reader = response.body?.getReader();
        if (!reader) {
          throw new ApiError(500, 'missing-stream', 'No response stream available');
        }

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split('\n\n');
          buffer = events.pop() ?? '';

          events.forEach((eventLine) => {
            const line = eventLine
              .split('\n')
              .find((segment) => segment.startsWith('data:'));
            if (!line) {
              return;
            }
            const afterPrefix = line.slice(5);
            const data = afterPrefix.startsWith(' ') ? afterPrefix.slice(1) : afterPrefix;
            if (data === '[DONE]') {
              get().actions.setResponseMs(Number.isFinite(headerMs) ? headerMs : null);
              return;
            }
            get().actions.appendToken(data);
          });
        }

        set({ isStreaming: false });
      } catch (error) {
        set({ isStreaming: false });
        const normalized =
          error instanceof ApiError
            ? error
            : new ApiError(500, 'unknown-request-id', (error as Error).message || 'Unknown error');
        get().actions.pushToast('error', `${normalized.message} (${normalized.requestId})`);
      }
    },
  },
}));
