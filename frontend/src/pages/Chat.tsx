import { useEffect, useMemo, useRef, useState, type FC, type KeyboardEvent } from 'react';
import { Link } from 'react-router-dom';

import MessageList from '../components/MessageList';
import SkillToggle from '../components/SkillToggle';
import { useChatStore, type SkillOption } from '../stores/chatStore';

const Chat: FC = () => {
  const messages = useChatStore((state) => state.messages);
  const skill = useChatStore((state) => state.skill);
  const modelId = useChatStore((state) => state.modelId);
  const models = useChatStore((state) => state.models);
  const modelsLoading = useChatStore((state) => state.modelsLoading);
  const isStreaming = useChatStore((state) => state.isStreaming);
  const inputError = useChatStore((state) => state.inputError);
  const actions = useChatStore((state) => state.actions);

  const [draft, setDraft] = useState<string>('');
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    void actions.initializeModels();
  }, [actions]);

  const isEmpty = useMemo(() => draft.trim().length === 0, [draft]);

  const updateTextareaSize = () => {
    if (!textareaRef.current) {
      return;
    }
    textareaRef.current.style.height = '0px';
    const nextHeight = Math.min(textareaRef.current.scrollHeight, 24 * 5);
    textareaRef.current.style.height = `${nextHeight}px`;
  };

  const handleSubmit = async () => {
    if (isStreaming || isEmpty) {
      return;
    }

    const content = draft;
    setDraft('');
    if (textareaRef.current) {
      textareaRef.current.style.height = '0px';
    }
    await actions.sendMessage(content);
  };

  const handleKeyDown = async (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      await handleSubmit();
    }
  };

  const handleSkillChange = (skillId: 'chat' | 'code') => {
    const nextSkill: SkillOption = {
      id: skillId,
      label: skillId === 'chat' ? 'Chat' : 'Code',
    };
    actions.setSkill(nextSkill);
    setDraft('');
  };

  const charCount = draft.length;
  const counterClass = charCount > 3800 ? 'text-red-400' : 'text-slate-400';

  return (
    <div className="flex h-screen flex-col bg-[radial-gradient(circle_at_20%_0%,rgba(56,189,248,0.15),transparent_45%),radial-gradient(circle_at_80%_100%,rgba(16,185,129,0.15),transparent_40%),#020617] text-slate-100 transition-all duration-300">
      <header className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-4 md:px-6">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-cyan-300">Local First AI</p>
          <h1 className="text-xl font-bold">Gemma Companion</h1>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Link
            to="/admin"
            className="rounded-lg border border-slate-500 px-3 py-2 text-xs font-semibold text-slate-200 transition duration-150 ease-in-out hover:border-cyan-300 hover:text-cyan-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300"
          >
            Admin
          </Link>
          <SkillToggle skill={skill.id} onChange={handleSkillChange} />
          <label className="sr-only" htmlFor="model-select">
            Select model
          </label>
          <select
            id="model-select"
            aria-label="Select model"
            value={modelId}
            disabled={isStreaming || modelsLoading}
            onChange={(event) => actions.setModel(event.target.value as 'gemma-2b' | 'gemma-e2b' | 'gemma-e4b')}
            className="rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {models.map((model) => (
              <option key={model.id} value={model.id} disabled={!model.available}>
                {model.label}
              </option>
            ))}
          </select>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col overflow-hidden rounded-t-3xl border border-slate-700/80 bg-slate-900/60 px-3 pb-0 pt-4 backdrop-blur md:px-6">
        <div className="flex-1 overflow-hidden">
          <MessageList messages={messages} isStreaming={isStreaming} />
        </div>

        <div className="sticky bottom-0 left-0 right-0 border-t border-slate-700 bg-slate-950/95 py-3 backdrop-blur">
          <div className="flex items-end gap-3">
            <button
              type="button"
              aria-label="Clear chat history"
              title="Clear history"
              onClick={actions.clearHistory}
              className="rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-200 transition duration-150 ease-in-out hover:border-cyan-400 hover:text-cyan-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300"
            >
              🧹
            </button>
            <div className="flex-1">
              <label htmlFor="chat-input" className="sr-only">
                Chat message input
              </label>
              <textarea
                id="chat-input"
                ref={textareaRef}
                aria-label="Message input"
                value={draft}
                onChange={(event) => {
                  setDraft(event.target.value);
                  updateTextareaSize();
                }}
                onKeyDown={handleKeyDown}
                placeholder="Ask something helpful..."
                rows={1}
                maxLength={4096}
                className="max-h-[120px] w-full resize-none rounded-xl border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-100 outline-none transition duration-150 ease-in-out placeholder:text-slate-500 focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/40"
              />
              <div className="mt-1 flex items-center justify-between">
                <span className={`text-xs ${counterClass}`}>{charCount} / 4096</span>
                {inputError ? <span className="text-xs text-red-400">{inputError}</span> : null}
              </div>
            </div>
            <button
              type="button"
              aria-label="Send message"
              disabled={isStreaming || isEmpty}
              onClick={() => {
                void handleSubmit();
              }}
              className="rounded-xl bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-900 transition duration-150 ease-in-out hover:bg-cyan-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Send
            </button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Chat;
