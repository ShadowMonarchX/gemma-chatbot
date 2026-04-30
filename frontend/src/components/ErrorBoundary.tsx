import { Component, type ErrorInfo, type ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  message: string;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  public constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      message: '',
    };
    this.handleRetry = this.handleRetry.bind(this);
  }

  public static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      message: error.message,
    };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // eslint-disable-next-line no-console
    console.error('React error boundary caught error', error, errorInfo);
  }

  public handleRetry(): void {
    this.setState({ hasError: false, message: '' });
  }

  public render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-slate-950 p-6 text-slate-100">
          <div className="w-full max-w-md rounded-2xl border border-red-500/40 bg-slate-900 p-6">
            <h1 className="text-xl font-semibold text-red-300">Something went wrong</h1>
            <p className="mt-3 text-sm text-slate-300">{this.state.message || 'Unexpected UI error.'}</p>
            <button
              type="button"
              aria-label="Retry rendering application"
              onClick={this.handleRetry}
              className="mt-5 rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-900 transition duration-150 ease-in-out hover:bg-cyan-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
