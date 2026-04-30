import { useEffect, useState, type FC } from 'react';

export interface SkillUsageBarProps {
  usage: Record<string, number>;
}

const SkillUsageBar: FC<SkillUsageBarProps> = ({ usage }) => {
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    const id = window.setTimeout(() => setAnimate(true), 40);
    return () => window.clearTimeout(id);
  }, [usage]);

  const total = Object.values(usage).reduce((sum, value) => sum + value, 0);
  const entries: Array<[string, number]> =
    Object.entries(usage).length > 0 ? Object.entries(usage) : [['chat', 0], ['code', 0]];

  return (
    <div className="space-y-3">
      {entries.map(([name, value]) => {
        const percent = total > 0 ? Math.round((value / total) * 100) : 0;
        return (
          <div key={name} className="space-y-1">
            <div className="flex items-center justify-between text-sm text-slate-200">
              <span className="font-medium">{name === 'chat' ? 'Chat' : 'Code'}</span>
              <span>{percent}%</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-slate-700">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400 transition-all duration-500 ease-out"
                style={{ width: animate ? `${percent}%` : '0%' }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default SkillUsageBar;
