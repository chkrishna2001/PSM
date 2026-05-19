export interface LocomoTurn {
  dia_id?: string;
  speaker?: string;
  text?: string;
  img_url?: string[];
  blip_caption?: string;
  query?: string;
  session?: string;
}

export interface LocomoQa {
  category?: string;
  question?: string;
  answer?: string;
  evidence?: string[];
}

export interface LocomoSample {
  sample_id?: string;
  conversation?: Record<string, LocomoTurn[] | string>;
  event_summary?: Record<string, { date?: string }>;
  session_summary?: Record<string, string>;
  qa?: LocomoQa[];
}

export interface CliOptions {
  data: string;
  db: string;
  server: string;
  out: string;
  limit: number;
  batchSize: number;
  topK: number;
}
