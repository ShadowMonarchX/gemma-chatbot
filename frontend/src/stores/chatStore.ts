import { create } from 'zustand';

import { ApiError, apiCall, apiStream } from '../api/client';
import {
  ChatRequestSchema,
  ModelsResponseSchema,
  type ChatRequest,
  type ModelId,
  type ModelInfo,
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

export interface ModelOption {
  id: ModelId;
  label: string;
  available: boolean;
  description: string;
  backend: string;
  quantization: string;
  aliasOf?: string;
}

interface ChatActions {
  initializeModels: () => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  setSkill: (skill: SkillOption) => void;
  setModel: (modelId: ModelId) => void;
  clearHistory: () => void;
  appendToken: (token: string) => void;
  setResponseMs: (ms: number | null) => void;
  pushToast: (kind: 'error' | 'success', message: string) => void;
  dismissToast: (id: string) => void;
}

interface ChatState {
  messages: UiMessage[];
  skill: SkillOption;
  modelId: ModelId;
  models: ModelOption[];
  modelsLoading: boolean;
  isStreaming: boolean;
  responseMs: number | null;
  inputError: string | null;
  toasts: ToastItem[];
  actions: ChatActions;
}

const DEFAULT_SKILL: SkillOption = { id: 'chat', label: 'Chat' };
const DEFAULT_MODEL: ModelId = 'gemma-2b';
const MAX_MESSAGE_LENGTH = 4096;

class ChatStoreFormatter {
  public toModelOptions(items: ModelInfo[]): ModelOption[] {
    return items.map((item) => ({
      id: item.id as ModelId,
      label: item.label,
      available: item.available,
      description: item.description,
      backend: item.backend,
      quantization: item.quantization,
      aliasOf: item.alias_of ?? undefined,
    }));
  }
}

const formatter = new ChatStoreFormatter();

export const useChatStore = create<ChatState>()((set, get) => ({
  messages: [],
  skill: DEFAULT_SKILL,
  modelId: DEFAULT_MODEL,
  models: [
    {
      id: 'gemma-2b',
      label: 'Gemma 2B',
      available: true,
      description: 'Fast default model.',
      backend: 'mlx',
      quantization: 'INT4',
    },
    {
      id: 'gemma-e2b',
      label: 'Gemma E2B',
      available: true,
      description: 'Efficient mode profile.',
      backend: 'mlx',
      quantization: 'INT4',
    },
    {
      id: 'gemma-e4b',
      label: 'Gemma E4B',
      available: true,
      description: 'Higher-quality mode profile.',
      backend: 'mlx',
      quantization: 'INT4',
    },
  ],
  modelsLoading: false,
  isStreaming: false,
  responseMs: null,
  inputError: null,
  toasts: [],
  actions: {
    initializeModels: async () => {
      set({ modelsLoading: true });
      try {
        const payload = await apiCall('/models', { method: 'GET' });
        const parsed = ModelsResponseSchema.parse(payload);
        const options = formatter.toModelOptions(parsed.models);
        const active = options.find((item) => item.id === parsed.active_model_id) ?? options[0];

        set({
          models: options,
          modelId: active ? active.id : DEFAULT_MODEL,
          modelsLoading: false,
        });
      } catch (error) {
        set({ modelsLoading: false });
        const normalized =
          error instanceof ApiError
            ? error
            : new ApiError(500, 'unknown-request-id', (error as Error).message || 'Unknown error');
        get().actions.pushToast('error', `${normalized.message} (${normalized.requestId})`);
      }
    },
    setSkill: (skill: SkillOption) => {
      set({
        skill,
        messages: [],
        isStreaming: false,
        responseMs: null,
        inputError: null,
      });
    },
    setModel: (modelId: ModelId) => {
      set({
        modelId,
        messages: [],
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
        modelId: state.modelId,
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
        model_id: state.modelId,
        stream: true,
      };

      const parsed = ChatRequestSchema.safeParse(requestPayload);
      if (!parsed.success) {
        const issue = parsed.error.issues[0];
        const message = issue ? issue.message : 'Invalid request payload.';
        set({ inputError: message });
        get().actions.pushToast('error', message);
        return;
      }

      set({
        messages: nextMessages,
        isStreaming: true,
        responseMs: null,
        inputError: null,
      });

      try {
        const startedAt = performance.now();
        const response = await apiStream('/chat', {
          method: 'POST',
          body: JSON.stringify(parsed.data),
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
            const dataLine = eventLine.split('\n').find((segment) => segment.startsWith('data:'));
            if (!dataLine) {
              return;
            }

            const withoutPrefix = dataLine.slice(5);
            const data = withoutPrefix.startsWith(' ') ? withoutPrefix.slice(1) : withoutPrefix;

            if (data === '[DONE]') {
              const fallbackMs = Math.max(Math.round(performance.now() - startedAt), 1);
              const resolvedMs =
                Number.isFinite(headerMs) && headerMs > 0 ? Math.round(headerMs) : fallbackMs;
              get().actions.setResponseMs(resolvedMs);
              return;
            }

            get().actions.appendToken(data);
          });
        }
      } catch (error) {
        const normalized =
          error instanceof ApiError
            ? error
            : new ApiError(500, 'unknown-request-id', (error as Error).message || 'Unknown error');
        get().actions.pushToast('error', `${normalized.message} (${normalized.requestId})`);
      } finally {
        set({ isStreaming: false });
      }
    },
  },
}));
