import { z } from 'zod';

export const ChatMessageSchema = z.object({
  role: z.enum(['user', 'assistant']),
  content: z.string().min(1).max(4096),
});

export const ChatRequestSchema = z.object({
  messages: z.array(ChatMessageSchema).min(1).max(20),
  skill_id: z.string().regex(/^[a-z_]+$/),
  model_id: z.string().regex(/^[a-z0-9_-]+$/),
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
  cuda_gpu: z.boolean(),
  is_apple_silicon: z.boolean(),
  platform_system: z.string(),
});

export const HealthResponseSchema = z.object({
  status: z.string(),
  model_id: z.string(),
  model_label: z.string(),
  backend: z.string(),
  quantization: z.string(),
  hardware: HardwareInfoSchema,
  model_load_ms: z.number(),
  avg_tokens_per_sec: z.number(),
  last_tokens_per_sec: z.number(),
  uptime_seconds: z.number(),
  last_request_ms: z.number(),
});

export const AdminResponseSchema = HealthResponseSchema.extend({
  total_requests: z.number(),
  errors: z.number(),
  avg_response_ms: z.number(),
  avg_first_token_ms: z.number(),
  requests_per_minute: z.number(),
  skill_usage: z.record(z.number()),
  model_usage: z.record(z.number()),
  injection_blocks: z.number(),
  rate_limit_hits: z.number(),
});

export const ModelInfoSchema = z.object({
  id: z.string(),
  label: z.string(),
  backend: z.string(),
  source: z.string(),
  quantization: z.string(),
  available: z.boolean(),
  default: z.boolean(),
  description: z.string(),
  alias_of: z.string().nullable().optional(),
});

export const ModelsResponseSchema = z.object({
  active_model_id: z.string(),
  models: z.array(ModelInfoSchema),
});

export const SkillListSchema = z.array(SkillSchema);

export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type ChatRequest = z.infer<typeof ChatRequestSchema>;
export type Skill = z.infer<typeof SkillSchema>;
export type HardwareInfo = z.infer<typeof HardwareInfoSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
export type AdminResponse = z.infer<typeof AdminResponseSchema>;
export type ModelInfo = z.infer<typeof ModelInfoSchema>;
export type ModelsResponse = z.infer<typeof ModelsResponseSchema>;

export type SkillId = 'chat' | 'code';
export type ModelId = 'gemma-2b' | 'gemma-e2b' | 'gemma-e4b';

export interface UiMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  skillId?: SkillId;
  modelId?: ModelId;
  responseMs?: number | null;
}
