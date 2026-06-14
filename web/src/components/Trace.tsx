import type { Trace } from "../types";
import { VariantCard } from "./VariantCard";

export function TracePanel({ trace, streaming }: { trace: Trace; streaming: boolean }) {
  if (!trace.modal && trace.variants.length === 0) {
    return (
      <div className="trace empty">
        <p>The process trace appears here: the modal (greedy) answer, the
        brainstormed variants, their novelty-vs-modal and quality scores, the
        Pareto frontier, and the chosen reply.</p>
      </div>
    );
  }
  return (
    <div className="trace">
      <h3>Process trace {streaming && <span className="pulse">live</span>}</h3>
      {trace.modal && (
        <div className="modal-box">
          <div className="trace-label">modal (greedy) answer — what it would have said</div>
          <div className="modal-text">{trace.modal}</div>
        </div>
      )}
      {trace.variants.length > 0 && (
        <>
          <div className="trace-label">brainstormed variants — selected for novelty + quality</div>
          <div className="variant-grid">
            {trace.variants.map((item, i) => (
              <VariantCard
                key={i}
                item={item}
                score={trace.scores[i]}
                onFrontier={trace.frontier[i] ?? false}
                chosen={trace.selected === i}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
