import type { FC } from 'react';

import type { SkillId } from '../api/types';

export interface SkillToggleProps {
  skill: SkillId;
  onChange: (skill: SkillId) => void;
}

const SkillToggle: FC<SkillToggleProps> = ({ skill, onChange }) => {
  return (
    <div className="inline-flex rounded-full border border-slate-500 bg-slate-800 p-1 shadow-sm">
      {(['chat', 'code'] as const).map((id) => {
        const active = id === skill;
        return (
          <button
            key={id}
            aria-label={`Switch to ${id} mode`}
            type="button"
            onClick={() => onChange(id)}
            className={`rounded-full px-4 py-1.5 text-sm font-medium transition-all duration-150 ease-in-out focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-400 ${
              active
                ? 'bg-cyan-400 text-slate-900'
                : 'text-slate-300 hover:bg-slate-700 hover:text-slate-100'
            }`}
          >
            {id === 'chat' ? 'Chat' : 'Code'}
          </button>
        );
      })}
    </div>
  );
};

export default SkillToggle;
