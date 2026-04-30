import { useCallback, useEffect, useMemo, useState } from "react";

type AdminPayload = {
  status: string;
  model: string;
  quantization: string;
  hardware: {
    chip: string;
    ram_total_gb: number;
    ram_available_gb: number;
    cpu_cores: number;
    metal_gpu: boolean;
  };
  model_load_ms: number;
  avg_tokens_per_sec: number;
  uptime_seconds: number;
  last_request_ms: number;
  total_requests: number;
  errors: number;
  avg_response_ms: number;
  skill_usage: Record<string, number>;
  hallucination_guards_triggered: number;
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";

const numberFormat = new Intl.NumberFormat("en-US");

function MetricCard({ label, value, accent }: { label: string; value: string; accent?: "ok" | "bad" }) {
  const accentClass =
    accent === "ok"
      ? "text-emerald-600 dark:text-emerald-400"
      : accent === "bad"
        ? "text-rose-600 dark:text-rose-400"
        : "text-slate-900 dark:text-slate-50";

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/80 p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900/70">
      <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
      <p className={`mt-2 font-heading text-xl font-semibold ${accentClass}`}>{value}</p>
    </div>
  );
}

export default function Admin() {
  const [data, setData] = useState<AdminPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadAdmin = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/admin`);
      if (!response.ok) {
        throw new Error(`Admin request failed with status ${response.status}`);
      }
      const payload = (await response.json()) as AdminPayload;
      setData(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadAdmin();
    const timer = window.setInterval(() => {
      void loadAdmin();
    }, 10000);

    return () => window.clearInterval(timer);
  }, [loadAdmin]);

  const healthState = useMemo(() => {
    if (!data) {
      return { label: "Degraded", ok: false };
    }

    const healthy = data.status === "ok" && data.errors === 0;
    return {
      label: healthy ? "All systems nominal" : "Degraded",
      ok: healthy
    };
  }, [data]);

  const chatCount = data?.skill_usage.chat ?? 0;
  const codeCount = data?.skill_usage.code ?? 0;
  const totalSkillCalls = chatCount + codeCount;
  const chatPct = totalSkillCalls === 0 ? 0 : (chatCount / totalSkillCalls) * 100;
  const codePct = totalSkillCalls === 0 ? 0 : (codeCount / totalSkillCalls) * 100;

  return (
    <section className="animate-fadeUp rounded-3xl border border-slate-200/70 bg-white/85 p-4 shadow-panel backdrop-blur-md dark:border-slate-700/60 dark:bg-slate-900/80 dark:shadow-panelDark sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-heading text-xl font-semibold text-slate-900 dark:text-slate-50">
            Admin - hardware health
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Live runtime metrics (auto-refresh every 10 seconds)
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            void loadAdmin();
          }}
          className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition hover:border-cyan-400 hover:text-cyan-700 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:hover:border-cyan-400 dark:hover:text-cyan-300"
        >
          Refresh
        </button>
      </div>

      <div className="mt-5">
        <div
          className={`inline-flex rounded-full px-4 py-2 text-sm font-semibold ${
            healthState.ok
              ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300"
              : "bg-rose-100 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300"
          }`}
        >
          {healthState.label}
        </div>
      </div>

      {error ? (
        <div className="mt-4 rounded-xl border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:border-rose-500/40 dark:bg-rose-500/10 dark:text-rose-200">
          {error}
        </div>
      ) : null}

      <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Chip" value={data?.hardware.chip ?? "-"} />
        <MetricCard
          label="RAM Available"
          value={
            data
              ? `${data.hardware.ram_available_gb.toFixed(1)} / ${data.hardware.ram_total_gb} GB`
              : "-"
          }
        />
        <MetricCard
          label="Quantization"
          value={(data?.quantization ?? "-").replace("-mlx", " · mlx").replace("-gguf", " · gguf")}
        />
        <MetricCard
          label="Metal GPU"
          value={data?.hardware.metal_gpu ? "Active" : "Inactive"}
          accent={data?.hardware.metal_gpu ? "ok" : "bad"}
        />
      </div>

      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Total Requests"
          value={data ? numberFormat.format(data.total_requests) : "-"}
        />
        <MetricCard
          label="Avg Response (ms)"
          value={data ? numberFormat.format(data.avg_response_ms) : "-"}
        />
        <MetricCard
          label="Avg Tokens/Sec"
          value={data ? `${data.avg_tokens_per_sec.toFixed(1)}` : "-"}
        />
        <MetricCard label="Errors" value={data ? `${data.errors}` : "-"} accent={data?.errors ? "bad" : "ok"} />
      </div>

      <div className="mt-6 rounded-2xl border border-slate-200 bg-white/75 p-4 dark:border-slate-700 dark:bg-slate-900/70">
        <div className="mb-3 flex items-center justify-between">
          <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">Skill usage</p>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            chat {chatCount} • code {codeCount}
          </p>
        </div>

        <div className="space-y-3">
          <div>
            <div className="mb-1 flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
              <span>Chat</span>
              <span>{chatPct.toFixed(0)}%</span>
            </div>
            <div className="h-3 rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className="h-3 rounded-full bg-gradient-to-r from-emerald-400 to-cyan-500"
                style={{ width: `${chatPct}%` }}
              />
            </div>
          </div>

          <div>
            <div className="mb-1 flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
              <span>Code</span>
              <span>{codePct.toFixed(0)}%</span>
            </div>
            <div className="h-3 rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className="h-3 rounded-full bg-gradient-to-r from-blue-400 to-indigo-500"
                style={{ width: `${codePct}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 text-xs text-slate-500 dark:text-slate-400">
        <span>{loading ? "Refreshing..." : "Up to date"}</span>
        <span className="mx-2">•</span>
        <span>Model: {data?.model ?? APP_MODEL_FALLBACK}</span>
      </div>
    </section>
  );
}

const APP_MODEL_FALLBACK = "gemma-4-2b-it";
