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
}

export type TraceEvent =
  | { type: "modal"; text: string }
  | { type: "variants"; items: VariantItem[] }
  | { type: "scored"; index: number; novelty: number; distance: number; quality: number }
  | { type: "selected"; index: number; frontier: boolean[] }
  | { type: "response"; text: string; index: number }
  | { type: "error"; message: string };

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface Config {
  k: number;
  novelty_weight: number;
  convergent_floor: number;
  temperature: number;
}

// Live trace assembled for the current turn.
export interface Trace {
  modal: string | null;
  variants: VariantItem[];
  scores: Record<number, Scored>;
  frontier: boolean[];
  selected: number | null;
  done: boolean;
}

export const emptyTrace = (): Trace => ({
  modal: null,
  variants: [],
  scores: {},
  frontier: [],
  selected: null,
  done: false,
});
