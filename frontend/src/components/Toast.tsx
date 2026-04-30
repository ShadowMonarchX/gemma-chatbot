import { useEffect, type FC } from 'react';

import type { ToastItem } from '../stores/chatStore';

export interface ToastProps {
  toasts: ToastItem[];
  onDismiss: (id: string) => void;
}

const Toast: FC<ToastProps> = ({ toasts, onDismiss }) => {
  useEffect(() => {
    const timers = toasts.map((toast) =>
      window.setTimeout(() => {
        onDismiss(toast.id);
      }, 4000)
    );

    return () => {
      timers.forEach((timer) => window.clearTimeout(timer));
    };
  }, [toasts, onDismiss]);

  return (
    <div className="pointer-events-none fixed right-4 top-4 z-50 flex w-80 flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`pointer-events-auto rounded-xl border px-4 py-3 text-sm shadow-lg transition-all duration-300 ${
            toast.kind === 'error'
              ? 'border-red-400/40 bg-red-500/90 text-white'
              : 'border-emerald-400/40 bg-emerald-500/90 text-slate-900'
          }`}
          role="status"
          aria-live="polite"
        >
          <div className="flex items-start justify-between gap-3">
            <p>{toast.message}</p>
            <button
              type="button"
              aria-label="Dismiss notification"
              onClick={() => onDismiss(toast.id)}
              className="rounded-md px-2 py-1 text-xs font-semibold hover:bg-black/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
            >
              Dismiss
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

export default Toast;
