import type { FC, ReactNode } from 'react';

export interface AdminCardProps {
  title: string;
  value: ReactNode;
  subtitle?: string;
}

const AdminCard: FC<AdminCardProps> = ({ title, value, subtitle }) => {
  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4 shadow-sm transition duration-150 ease-in-out hover:border-cyan-400/50">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">{title}</h3>
      <div className="mt-3 text-2xl font-bold text-slate-100">{value}</div>
      {subtitle ? <p className="mt-1 text-xs text-slate-400">{subtitle}</p> : null}
    </div>
  );
};

export default AdminCard;
