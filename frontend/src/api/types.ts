import { z } from 'zod';

export const ChatMessageSchema = z.object({
  role: z.enum(['user', 'assistant']),
  content: z.string().min(1).max(4096),
});

export const ChatRequestSchema = z.object({
  messages: z.array(ChatMessageSchema).min(1).max(20),
  skill_id: z.string().regex(/^[a-z_]+$/),
  stream: z.boolean(),
});

export const SkillSchema = z.object({
  id: z.string(),
  label: z.string(),
  system_prompt: z.string(),
});

export const HardwareInfoSchema = z.object({
  chip: z.string(),
  ram_total_gb: z.number(),
  ram_available_gb: z.number(),
  cpu_cores: z.number(),
  metal_gpu: z.boolean(),
});

export const HealthResponseSchema = z.object({
  status: z.string(),
  model: z.string(),
  quantization: z.string(),
  hardware: HardwareInfoSchema,
  model_load_ms: z.number(),
  avg_tokens_per_sec: z.number(),
  uptime_seconds: z.number(),
  last_request_ms: z.number(),
});

export const AdminResponseSchema = HealthResponseSchema.extend({
  total_requests: z.number(),
  errors: z.number(),
  avg_response_ms: z.number(),
  requests_per_minute: z.number(),
  skill_usage: z.record(z.number()),
  hallucination_guards_triggered: z.number(),
  rate_limit_hits: z.number(),
});

export const SkillListSchema = z.array(SkillSchema);

export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type ChatRequest = z.infer<typeof ChatRequestSchema>;
export type Skill = z.infer<typeof SkillSchema>;
export type HardwareInfo = z.infer<typeof HardwareInfoSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
export type AdminResponse = z.infer<typeof AdminResponseSchema>;

export type SkillId = 'chat' | 'code';

export interface UiMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  skillId?: SkillId;
  responseMs?: number | null;
}
