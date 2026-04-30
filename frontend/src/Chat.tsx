import { useEffect, useMemo, useRef, useState } from "react";
import type { Skill } from "./skills";

type Role = "user" | "assistant";

type UiMessage = {
  id: string;
  role: Role;
  content: string;
  responseMs?: number;
};

type Props = {
  skills: Skill[];
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";

const makeId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

export default function Chat({ skills }: Props) {
  const [selectedSkillId, setSelectedSkillId] = useState(skills[0]?.id ?? "chat");
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorText, setErrorText] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  const selectedSkill = useMemo(
    () => skills.find((skill) => skill.id === selectedSkillId) ?? skills[0],
    [selectedSkillId, skills]
  );

  useEffect(() => {
    scrollerRef.current?.scrollTo({
      top: scrollerRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [messages, isStreaming]);

  useEffect(() => {
    setMessages([]);
    setErrorText(null);
    setInput("");
  }, [selectedSkillId]);

  const sendMessage = async () => {
    const content = input.trim();
    if (!content || isStreaming || !selectedSkill) {
      return;
    }

    const userMsg: UiMessage = { id: makeId(), role: "user", content };
    const assistantId = makeId();
    const assistantMsg: UiMessage = { id: assistantId, role: "assistant", content: "" };

    const conversation = [...messages, userMsg];
    setMessages([...conversation, assistantMsg]);
    setInput("");
    setErrorText(null);
    setIsStreaming(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          messages: conversation.map((msg) => ({ role: msg.role, content: msg.content })),
          skill: selectedSkill.id,
          stream: true
        })
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(body || `Request failed with status ${response.status}`);
      }

      const headerMs = Number(response.headers.get("x-response-ms") ?? "0");
      if (!response.body) {
        throw new Error("Missing streaming response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let assistantText = "";
      let doneMs: number | undefined;

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const eventBlock of events) {
          const lines = eventBlock.split(/\r?\n/);
          let eventType = "message";
          let data = "";

          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              data += line.slice(5).trim();
            }
          }

          if (!data) {
            continue;
          }

          const parsed = JSON.parse(data) as { token?: string; response_ms?: number };

          if (eventType === "token" && typeof parsed.token === "string") {
            assistantText += parsed.token;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? {
                      ...msg,
                      content: assistantText
                    }
                  : msg
              )
            );
          }

          if (eventType === "done" && typeof parsed.response_ms === "number") {
            doneMs = parsed.response_ms;
          }
        }
      }

      const finalMs = doneMs ?? headerMs;
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                responseMs: Number.isFinite(finalMs) ? finalMs : 0
              }
            : msg
        )
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setErrorText(message);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content: "I am not sure. The backend request failed.",
                responseMs: 0
              }
            : msg
        )
      );
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <section className="animate-fadeUp rounded-3xl border border-slate-200/70 bg-white/85 shadow-panel backdrop-blur-md dark:border-slate-700/60 dark:bg-slate-900/80 dark:shadow-panelDark">
      <div className="flex items-center justify-between gap-4 border-b border-slate-200/80 px-4 py-4 dark:border-slate-700/70 sm:px-6">
        <div>
          <p className="font-heading text-lg font-semibold tracking-tight text-slate-900 dark:text-slate-50">
            Gemma 4 Local Chatbot
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400">Local-first, no auth, hardware-aware runtime</p>
        </div>

        <div className="flex items-center rounded-full bg-slate-100 p-1 dark:bg-slate-800">
          {skills.map((skill) => (
            <button
              key={skill.id}
              type="button"
              onClick={() => setSelectedSkillId(skill.id)}
              className={`rounded-full px-3 py-1.5 text-xs font-semibold transition sm:px-4 sm:text-sm ${
                selectedSkillId === skill.id
                  ? "bg-gradient-to-r from-cyan-500 to-blue-500 text-white"
                  : "text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-slate-100"
              }`}
            >
              {skill.label}
            </button>
          ))}
        </div>
      </div>

      <div
        ref={scrollerRef}
        className="max-h-[56vh] min-h-[48vh] space-y-4 overflow-y-auto px-4 py-5 sm:px-6"
      >
        {messages.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/80 px-4 py-8 text-center text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900/30 dark:text-slate-400">
            Start chatting with <span className="font-semibold">{selectedSkill?.label}</span>. Skill changes clear
            the history automatically.
          </div>
        ) : null}

        {messages.map((message) => {
          const isUser = message.role === "user";
          return (
            <div
              key={message.id}
              className={`flex ${isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed sm:max-w-[75%] ${
                  isUser
                    ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                    : "border border-slate-200 bg-white text-slate-800 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                }`}
              >
                <div className="whitespace-pre-wrap">{message.content}</div>
                {!isUser && typeof message.responseMs === "number" ? (
                  <div className="mt-2 inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 font-mono text-[11px] text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                    {`\u23f1 ${Math.round(message.responseMs)} ms`}
                  </div>
                ) : null}
              </div>
            </div>
          );
        })}

        {isStreaming ? (
          <div className="flex justify-start">
            <div className="inline-flex items-center gap-1.5 rounded-2xl border border-slate-200 bg-white px-4 py-3 dark:border-slate-700 dark:bg-slate-800">
              <span className="h-2 w-2 animate-pulseDot rounded-full bg-cyan-500" />
              <span className="h-2 w-2 animate-pulseDot rounded-full bg-cyan-500 [animation-delay:160ms]" />
              <span className="h-2 w-2 animate-pulseDot rounded-full bg-cyan-500 [animation-delay:320ms]" />
            </div>
          </div>
        ) : null}
      </div>

      <div className="border-t border-slate-200/80 px-4 py-4 dark:border-slate-700/70 sm:px-6">
        {errorText ? (
          <p className="mb-2 text-xs text-rose-600 dark:text-rose-400">{errorText}</p>
        ) : null}

        <div className="flex items-end gap-3">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void sendMessage();
              }
            }}
            rows={2}
            placeholder={`Ask using ${selectedSkill?.label} skill...`}
            className="w-full resize-none rounded-2xl border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-200 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:border-cyan-400 dark:focus:ring-cyan-600/40"
          />
          <button
            type="button"
            onClick={() => {
              void sendMessage();
            }}
            disabled={isStreaming || !input.trim()}
            className="rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2 text-sm font-semibold text-white transition hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Send
          </button>
        </div>
      </div>
    </section>
  );
}
