export interface LocomoTurn {
  dia_id?: string;
  speaker?: string;
  text?: string;
}

export interface LocomoQa {
  category?: string;
  question?: string;
  answer?: string;
  evidence?: string[];
}

export interface LocomoSample {
  sample_id?: string;
  conversation?: Record<string, LocomoTurn[]>;
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
