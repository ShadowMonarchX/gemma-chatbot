import { useEffect, useMemo, useState, type FC } from 'react';

import type { UiMessage } from '../api/types';

interface ContentSegmentText {
  type: 'text';
  value: string;
}

interface ContentSegmentCode {
  type: 'code';
  language: string;
  value: string;
}

type ContentSegment = ContentSegmentText | ContentSegmentCode;

export interface MessageBubbleProps {
  message: UiMessage;
}

declare global {
  interface Window {
    Prism?: {
      highlightAllUnder: (element: ParentNode) => void;
    };
  }
}

class MessageLabelFormatter {
  public skillLabel(skillId: string | undefined): string {
    if (skillId === 'code') {
      return 'Code';
    }
    return 'Chat';
  }

  public modelLabel(modelId: string | undefined): string {
    if (modelId === 'gemma-e2b') {
      return 'E2B';
    }
    if (modelId === 'gemma-e4b') {
      return 'E4B';
    }
    return '2B';
  }
}

const formatter = new MessageLabelFormatter();

const MessageBubble: FC<MessageBubbleProps> = ({ message }) => {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const segments = useMemo<ContentSegment[]>(() => {
    const parsedSegments: ContentSegment[] = [];
    const regex = /```([a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g;
    let currentIndex = 0;
    let match = regex.exec(message.content);

    while (match) {
      const fullMatch = match[0];
      const language = match[1] ?? 'plaintext';
      const code = match[2] ?? '';
      const before = message.content.slice(currentIndex, match.index);
      if (before.trim().length > 0) {
        parsedSegments.push({ type: 'text', value: before });
      }
      parsedSegments.push({ type: 'code', language, value: code });
      currentIndex = match.index + fullMatch.length;
      match = regex.exec(message.content);
    }

    const remainder = message.content.slice(currentIndex);
    if (remainder.trim().length > 0 || parsedSegments.length === 0) {
      parsedSegments.push({ type: 'text', value: remainder });
    }

    return parsedSegments;
  }, [message.content]);

  useEffect(() => {
    const styleId = 'prism-theme-css';
    if (!document.getElementById(styleId)) {
      const link = document.createElement('link');
      link.id = styleId;
      link.rel = 'stylesheet';
      link.href = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css';
      document.head.appendChild(link);
    }

    const scriptId = 'prism-js';
    if (!document.getElementById(scriptId)) {
      const script = document.createElement('script');
      script.id = scriptId;
      script.src = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js';
      script.async = true;
      script.onload = () => {
        if (window.Prism) {
          window.Prism.highlightAllUnder(document.body);
        }
      };
      document.body.appendChild(script);
      return;
    }

    if (window.Prism) {
      window.Prism.highlightAllUnder(document.body);
    }
  }, [message.content]);

  const isUser = message.role === 'user';

  const handleCopy = async (value: string, key: string) => {
    await navigator.clipboard.writeText(value);
    setCopiedKey(key);
    window.setTimeout(() => {
      setCopiedKey((current) => (current === key ? null : current));
    }, 2000);
  };

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[92%] rounded-2xl border px-4 py-3 shadow-sm transition-all duration-150 ease-in-out md:max-w-[75%] ${
          isUser
            ? 'border-slate-900 bg-slate-900 text-slate-100'
            : 'border-slate-600 bg-slate-800/80 text-slate-100'
        }`}
      >
        <div className="space-y-3">
          {segments.map((segment, index) => {
            if (segment.type === 'text') {
              return (
                <p key={`text-${index}`} className="whitespace-pre-wrap text-sm leading-relaxed">
                  {segment.value}
                </p>
              );
            }

            const codeKey = `code-${index}`;
            return (
              <div key={codeKey} className="overflow-hidden rounded-xl border border-slate-600 bg-slate-900">
                <div className="flex items-center justify-between border-b border-slate-700 px-3 py-2 text-xs">
                  <span className="uppercase tracking-wide text-slate-300">{segment.language}</span>
                  <button
                    type="button"
                    aria-label="Copy code block"
                    onClick={() => {
                      void handleCopy(segment.value, codeKey);
                    }}
                    className="rounded-md bg-cyan-400/90 px-2 py-1 font-medium text-slate-900 transition duration-150 ease-in-out hover:bg-cyan-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300"
                  >
                    {copiedKey === codeKey ? 'Copied!' : 'Copy'}
                  </button>
                </div>
                <pre className="overflow-x-auto p-3 text-sm">
                  <code className={`language-${segment.language}`}>{segment.value}</code>
                </pre>
              </div>
            );
          })}
        </div>

        {!isUser ? (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-300">
            <span className="rounded-full border border-slate-500 bg-slate-700 px-2 py-0.5">
              {formatter.skillLabel(message.skillId)}
            </span>
            <span className="rounded-full border border-cyan-500/40 bg-cyan-500/10 px-2 py-0.5 text-cyan-200">
              {formatter.modelLabel(message.modelId)}
            </span>
            {typeof message.responseMs === 'number' ? (
              <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-emerald-200">
                ⏱ {message.responseMs} ms
              </span>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default MessageBubble;
