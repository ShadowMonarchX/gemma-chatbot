import { useEffect, type FC } from 'react';

import AdminCard from '../components/AdminCard';
import SkillUsageBar from '../components/SkillUsageBar';
import { useAdminStore } from '../stores/adminStore';

const formatUptime = (uptimeSeconds: number): string => {
  const hours = Math.floor(uptimeSeconds / 3600);
  const minutes = Math.floor((uptimeSeconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
};

const Admin: FC = () => {
  const data = useAdminStore((state) => state.data);
  const loading = useAdminStore((state) => state.loading);
  const error = useAdminStore((state) => state.error);
  const lastUpdated = useAdminStore((state) => state.lastUpdated);
  const actions = useAdminStore((state) => state.actions);

  useEffect(() => {
    void actions.fetchAdmin();
    const intervalId = window.setInterval(() => {
      void actions.fetchAdmin();
    }, 10000);
    return () => window.clearInterval(intervalId);
  }, [actions]);

  if (loading && !data) {
    return (
      <div className="min-h-screen bg-[linear-gradient(160deg,#020617_0%,#0f172a_45%,#022c22_100%)] px-4 py-6 text-slate-100 md:px-6">
        <div className="mx-auto max-w-6xl">
          <div className="mb-6 h-10 w-64 animate-pulse rounded bg-slate-700" />
          <div className="grid gap-4 md:grid-cols-2">
            {Array.from({ length: 8 }).map((_, index) => (
              <div key={String(index)} className="h-28 animate-pulse rounded-2xl bg-slate-800" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[linear-gradient(160deg,#020617_0%,#0f172a_45%,#022c22_100%)] p-6 text-slate-100">
        <div className="w-full max-w-lg rounded-2xl border border-red-400/40 bg-slate-900 p-6">
          <h2 className="text-lg font-semibold text-red-300">Admin data unavailable</h2>
          <p className="mt-2 text-sm text-slate-300">{error}</p>
          <button
            type="button"
            aria-label="Retry loading admin data"
            onClick={() => {
              void actions.fetchAdmin();
            }}
            className="mt-4 rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-900 transition duration-150 ease-in-out hover:bg-cyan-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  const isHealthy = data.status === 'ok' && data.errors === 0;
  const statusLabel = isHealthy ? 'All systems nominal' : 'System degraded';

  return (
    <div className="min-h-screen bg-[linear-gradient(160deg,#020617_0%,#0f172a_45%,#022c22_100%)] px-4 py-6 text-slate-100 transition-all duration-300 md:px-6">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold">Admin - hardware health</h1>
            <p className="text-sm text-slate-300">
              Last updated:{' '}
              {lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : 'not available'}
            </p>
          </div>
          <button
            type="button"
            aria-label="Refresh admin metrics"
            onClick={() => {
              void actions.fetchAdmin();
            }}
            className="rounded-lg border border-slate-500 bg-slate-800 px-4 py-2 text-sm font-semibold text-slate-100 transition duration-150 ease-in-out hover:border-cyan-300 hover:text-cyan-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300"
          >
            Refresh
          </button>
        </header>

        <section
          className={`rounded-2xl border px-4 py-3 text-sm font-medium ${
            isHealthy
              ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-200'
              : 'border-red-500/50 bg-red-500/10 text-red-200'
          }`}
        >
          {statusLabel}
        </section>

        <section className="grid gap-4 md:grid-cols-2">
          <AdminCard title="Chip" value={data.hardware.chip} />
          <AdminCard
            title="RAM"
            value={`${data.hardware.ram_available_gb.toFixed(1)} / ${data.hardware.ram_total_gb.toFixed(1)} GB`}
          />
          <AdminCard
            title="Quantization"
            value={data.quantization}
            subtitle={data.model.includes('gemma') ? 'Gemma runtime active' : data.model}
          />
          <AdminCard
            title="Metal GPU"
            value={data.hardware.metal_gpu ? 'Active' : 'Inactive'}
            subtitle={data.hardware.metal_gpu ? 'Acceleration enabled' : 'CPU fallback'}
          />
        </section>

        <section className="grid gap-4 md:grid-cols-2">
          <AdminCard title="Total requests" value={data.total_requests} />
          <AdminCard title="Avg response" value={`${Math.round(data.avg_response_ms)} ms`} />
          <AdminCard title="Tokens/sec" value={data.avg_tokens_per_sec.toFixed(1)} />
          <AdminCard title="Errors" value={data.errors} />
        </section>

        <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Skill usage</h2>
          <div className="mt-3">
            <SkillUsageBar usage={data.skill_usage} />
          </div>
        </section>

        <section className="rounded-xl border border-slate-700 bg-slate-900/70 px-4 py-3 text-xs text-slate-300">
          Injection attempts blocked: {data.hallucination_guards_triggered} | Rate limit hits:{' '}
          {data.rate_limit_hits} | Uptime: {formatUptime(data.uptime_seconds)}
        </section>
      </div>
    </div>
  );
};

export default Admin;
