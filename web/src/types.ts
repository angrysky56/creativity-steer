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
  // Real semantic entropy (paper Eq. 4) over the candidate pool.
  semantic_entropy?: number;
  norm_entropy?: number;
  num_clusters?: number;
  num_candidates?: number;
  prob_weighted?: boolean;
  basin_escape?: number;
  cluster_ids?: number[];
  // Connected-chain telemetry.
  trajectory?: boolean;
  trajectory_waves?: number;
  refine_passes?: number;
  refine_accepted?: number;
  refine_collapsed?: number;
  refine_total?: number;
}

export interface GroundingInfo {
  memory: number;
  tools: number;
  snippets: string[];
}

export type TraceEvent =
  | { type: "modal"; text: string }
  | { type: "variants"; items: VariantItem[] }
  | {
      type: "scored";
      index: number;
      novelty: number;
      distance: number;
      quality: number;
      scores?: Record<string, number>;
    }
  | {
      type: "selected";
      index: number;
      frontier: boolean[];
      floor_met?: boolean;
      chosen_quality?: number;
    }
  | { type: "response"; text: string; index: number; synthesized?: boolean }
  | {
      type: "controller";
      rounds: number;
      diversity: number;
      final_temperature: number;
      breadth: number;
      primes: number;
      branched: boolean;
      weights: Record<string, number>;
      quality_floor: number;
      semantic_entropy?: number;
      norm_entropy?: number;
      num_clusters?: number;
      num_candidates?: number;
      prob_weighted?: boolean;
      basin_escape?: number;
      cluster_ids?: number[];
      trajectory?: boolean;
      trajectory_waves?: number;
      refine_passes?: number;
      refine_accepted?: number;
      refine_collapsed?: number;
      refine_total?: number;
    }
  | {
      type: "chain";
      trajectory_waves: { wave: number; pool: number; clusters: number }[];
      refine: {
        pass: number;
        prime: number;
        axis: string;
        accepted: boolean;
        collapsed: boolean;
        old: number;
        new: number;
      }[];
    }
  | { type: "synthesis"; sources: number; collapsed_to_modal?: boolean }
  | { type: "grounding"; memory: number; tools: number; snippets: string[] }
  | { type: "error"; message: string };

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  trace?: Trace;
  config?: Config;
}

export interface Config {
  k: number;
  seed?: number; // 0 = random; any other value reproduces the run
  novelty_weight: number;
  convergent_floor: number;
  temperature: number;
  coherence_weight: number;
  breadth_k: number;
  prime_n: number;
  branch: boolean;
  synthesize: boolean;
  openness_weight: number;
  openness_branches: number;
  originality_weight?: number; // freshness vs. recognised clichés (judge axis)
  surprise_weight?: number; // model token-confidence (recitation vs. composition)
  // Connected-chain controls (optional; off unless a profile/slider sets them).
  trajectory?: boolean;
  refine_passes?: number;
  // Generation-length budgets (tokens). Optional: when omitted the server uses
  // its .env defaults. 0 = unlimited (the model's own EOS ends the reply).
  response_tokens?: number;
  brainstorm_tokens?: number;
  modal_tokens?: number;
  branch_tokens?: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  config: Config;
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
  floorMet: boolean;
  chosenQuality: number | null;
  synthesisCollapsed: boolean;
  grounding: GroundingInfo | null;
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
  floorMet: true,
  chosenQuality: null,
  synthesisCollapsed: false,
  grounding: null,
});
