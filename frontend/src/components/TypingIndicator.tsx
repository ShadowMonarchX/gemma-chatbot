import type { FC } from 'react';

export interface TypingIndicatorProps {
  visible: boolean;
}

const TypingIndicator: FC<TypingIndicatorProps> = ({ visible }) => {
  if (!visible) {
    return null;
  }

  return (
    <div className="mt-2 inline-flex items-center gap-1 rounded-full bg-slate-700 px-3 py-2 text-slate-200">
      <span className="sr-only">Assistant is typing</span>
      <span className="h-2 w-2 animate-bounce rounded-full bg-cyan-300 [animation-delay:-0.3s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-cyan-300 [animation-delay:-0.15s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-cyan-300" />
    </div>
  );
};

export default TypingIndicator;
