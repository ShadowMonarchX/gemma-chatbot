import { Navigate, Route, Routes } from 'react-router-dom';
import type { FC } from 'react';

import ErrorBoundary from './components/ErrorBoundary';
import Toast from './components/Toast';
import Admin from './pages/Admin';
import Chat from './pages/Chat';
import { useChatStore } from './stores/chatStore';

const ToastProvider: FC = () => {
  const toasts = useChatStore((state) => state.toasts);
  const dismissToast = useChatStore((state) => state.actions.dismissToast);

  return <Toast toasts={toasts} onDismiss={dismissToast} />;
};

const App: FC = () => {
  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/admin" element={<Admin />} />
      </Routes>
      <ToastProvider />
    </ErrorBoundary>
  );
};

export default App;
