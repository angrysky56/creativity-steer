// Event + state types shared across the UI.

export interface VariantItem {
  text: string;
  is_modal: boolean;
}

export interface Scored {
  index: number;
  novelty: number;
  distance: number;
  quality: number;
  scores?: Record<string, number>;
}

export interface ControllerInfo {
  rounds: number;
  diversity: number;
  final_temperature: number;
  breadth: number;
  primes: number;
  branched: boolean;
  weights: Record<string, number>;
  quality_floor: number;
}

export type TraceEvent =
  | { type: "modal"; text: string }
  | { type: "variants"; items: VariantItem[] }
  | { type: "scored"; index: number; novelty: number; distance: number; quality: number; scores?: Record<string, number> }
  | { type: "selected"; index: number; frontier: boolean[] }
  | { type: "response"; text: string; index: number; synthesized?: boolean }
  | { type: "controller"; rounds: number; diversity: number; final_temperature: number; breadth: number; primes: number; branched: boolean; weights: Record<string, number>; quality_floor: number }
  | { type: "synthesis"; sources: number }
  | { type: "error"; message: string };

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  trace?: Trace;
}

export interface Config {
  k: number;
  novelty_weight: number;
  convergent_floor: number;
  temperature: number;
  coherence_weight: number;
  breadth_k: number;
  prime_n: number;
  branch: boolean;
  synthesize: boolean;
}

// Live trace assembled for the current turn.
export interface Trace {
  modal: string | null;
  variants: VariantItem[];
  scores: Record<number, Scored>;
  frontier: boolean[];
  selected: number | null;
  done: boolean;
  controller: ControllerInfo | null;
  synthesisSources: number | null;
  synthesized: boolean;
}

export const emptyTrace = (): Trace => ({
  modal: null,
  variants: [],
  scores: {},
  frontier: [],
  selected: null,
  done: false,
  controller: null,
  synthesisSources: null,
  synthesized: false,
});
