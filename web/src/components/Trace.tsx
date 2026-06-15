import { useState } from "react";
import type { Trace } from "../types";
import { ParetoPlot } from "./ParetoPlot";
import { VariantCard } from "./VariantCard";

interface TracePanelProps {
  trace: Trace;
  streaming: boolean;
}

export function TracePanel({ trace, streaming }: TracePanelProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  if (!trace.modal && trace.variants.length === 0) {
    return (
      <div className="trace empty-state">
        <div className="empty-icon">📈</div>
        <h3>Decision Trace Analytics</h3>
        <p>
          Submit a message to view the real-time decision process trace.
        </p>
        <p className="empty-sub">
          The pipeline will brainstorm candidate answers, score their quality
          using a rubric judge, measure their semantic novelty relative to the
          greedy baseline, and pick the Pareto-optimal choice.
        </p>
        <div className="pipeline-steps-guide">
          <div className="guide-step">
            <span className="step-num">1</span>
            <span>Generate greedy modal baseline</span>
          </div>
          <div className="guide-step">
            <span className="step-num">2</span>
            <span>Brainstorm <i>k</i> diverse alternative angles</span>
          </div>
          <div className="guide-step">
            <span className="step-num">3</span>
            <span>Score novelty (Semantic Entropy) & quality</span>
          </div>
          <div className="guide-step">
            <span className="step-num">4</span>
            <span>Pareto selection above quality floor</span>
          </div>
        </div>
      </div>
    );
  }

  // Count how many variants have been scored
  const totalVariants = trace.variants.length;
  const scoredCount = Object.keys(trace.scores).length;
  const scoringProgress = totalVariants > 0 ? (scoredCount / totalVariants) * 100 : 0;

  return (
    <div className="trace">
      <div className="trace-header">
        <div>
          <h3>Decision Analytics</h3>
          <p className="trace-subtitle">Novelty vs. Quality Tradeoff</p>
        </div>
        {streaming ? (
          <span className="status-badge streaming">
            <span className="pulse-dot"></span>
            Steering Turn
          </span>
        ) : (
          <span className="status-badge done">Trace Ready</span>
        )}
      </div>

      {/* Progress bar while scoring candidates */}
      {streaming && totalVariants > 0 && scoredCount < totalVariants && (
        <div className="scoring-progress-bar">
          <div className="progress-label">
            <span>Judging candidates...</span>
            <span>{scoredCount} / {totalVariants}</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${scoringProgress}%` }}></div>
          </div>
        </div>
      )}

      {/* Visual Pareto Frontier Plot */}
      {scoredCount > 0 && (
        <div className="plot-section card-container">
          <div className="section-title">Pareto Frontier Space</div>
          <ParetoPlot
            trace={trace}
            hoveredIndex={hoveredIndex}
            setHoveredIndex={setHoveredIndex}
          />
          <div className="plot-legend">
            <span className="legend-item"><span className="dot modal-dot"></span>Modal</span>
            <span className="legend-item"><span className="dot frontier-dot"></span>Frontier</span>
            <span className="legend-item"><span className="dot chosen-dot"></span>Chosen</span>
            <span className="legend-item"><span className="dot other-dot"></span>Candidate</span>
          </div>
        </div>
      )}

      {trace.modal && (
        <div className="modal-box-section card-container">
          <div className="section-title">Baseline Reference Answer</div>
          <div className="modal-box">
            <div className="modal-text">{trace.modal}</div>
          </div>
          <div className="modal-note">
            The greedy/modal response is scored as having 0.00 novelty, acting as the distance reference for all brainstormed candidates.
          </div>
        </div>
      )}

      {trace.variants.length > 0 && (
        <div className="variants-section">
          <div className="section-title">Brainstormed Candidate Pool ({totalVariants})</div>
          <div className="variant-grid">
            {trace.variants.map((item, i) => (
              <VariantCard
                key={i}
                item={item}
                score={trace.scores[i]}
                onFrontier={trace.frontier[i] ?? false}
                chosen={trace.selected === i}
                isHovered={hoveredIndex === i}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
