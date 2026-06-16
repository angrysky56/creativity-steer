import { useState } from "react";
import type { Trace, Config } from "../types";
import { ParetoPlot } from "./ParetoPlot";
import { VariantCard } from "./VariantCard";
import { getActiveProfileKey } from "../App";

interface TracePanelProps {
  trace: Trace;
  streaming: boolean;
  configUsed?: Config | null;
  onApplyConfig?: (cfg: Config) => void;
}

export function TracePanel({ trace, streaming, configUsed, onApplyConfig }: TracePanelProps) {
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

      {/* Grounding Context Card */}
      {trace.grounding && (
        <div className="grounding-card">
          <div className="grounding-header-row">
            <span className="grounding-title">
              <span className="grounding-title-icon">📚</span>
              Grounding Context (Chroma DB + Tools)
            </span>
            <div className="grounding-badges">
              {trace.grounding.memory > 0 && (
                <span className="grounding-badge memories">
                  {trace.grounding.memory} Memories
                </span>
              )}
              {trace.grounding.tools > 0 && (
                <span className="grounding-badge tools">
                  {trace.grounding.tools} Tools
                </span>
              )}
              {trace.grounding.memory === 0 && trace.grounding.tools === 0 && (
                <span className="grounding-badge done">No Context</span>
              )}
            </div>
          </div>
          {trace.grounding.snippets && trace.grounding.snippets.length > 0 && (
            <div className="grounding-details">
              <details>
                <summary className="grounding-summary">Inspect Grounding Snippets</summary>
                <div className="grounding-snippets-list">
                  {trace.grounding.snippets.map((snip, index) => (
                    <div key={index} className="grounding-snippet">
                      {snip}
                    </div>
                  ))}
                </div>
              </details>
            </div>
          )}
        </div>
      )}

      {/* Self-Tuning Controller Card */}
      {trace.controller && (
        <div className="controller-status-card">
          <div className="controller-header-row">
            <span className="controller-title">
              <span className="controller-title-icon">⚙️</span>
              Self-Tuning Controller
            </span>
            <span className="controller-badge">Active</span>
          </div>
          <div className="metrics-grid">
            <div className="metric-box">
              <span className="metric-box-label">Exploration Rounds</span>
              <span className="metric-box-val">{trace.controller.rounds}</span>
            </div>
            <div
              className="metric-box"
              title="Rao-Blackwellised semantic entropy (nats) over the candidate pool, clustered by bidirectional entailment. Falls when many candidates share one meaning (mode collapse)."
            >
              <span className="metric-box-label">Semantic Entropy</span>
              <span className="metric-box-val">
                {(trace.controller.semantic_entropy ?? 0).toFixed(2)}
                <span className="metric-box-sub">
                  {" "}/ {(trace.controller.num_candidates
                    ? Math.log(trace.controller.num_candidates)
                    : 0).toFixed(2)} nats
                </span>
              </span>
            </div>
            <div
              className="metric-box"
              title="Distinct semantic classes among the candidates. Fewer than the candidate count means some ideas are paraphrases of each other."
            >
              <span className="metric-box-label">Semantic Clusters</span>
              <span className="metric-box-val">
                {trace.controller.num_clusters ?? "—"}
                <span className="metric-box-sub"> / {trace.controller.num_candidates ?? trace.variants.length}</span>
              </span>
            </div>
            <div className="metric-box">
              <span className="metric-box-label">Final Temp</span>
              <span className="metric-box-val">{trace.controller.final_temperature.toFixed(2)}</span>
            </div>
          </div>
          <div className="controller-details-row">
            <span className="controller-detail-item">
              {trace.controller.trajectory ? "Trajectory" : "Breadth"} Funnel:{" "}
              <strong>{trace.controller.breadth} candidates</strong> → <strong>{trace.controller.primes || trace.variants.length} primes</strong>
              {trace.controller.trajectory && trace.controller.trajectory_waves
                ? ` (${trace.controller.trajectory_waves} waves)`
                : ""}
            </span>
            <span className="controller-detail-item">
              Quality Floor: <strong>{trace.controller.quality_floor.toFixed(2)}</strong>
            </span>
            {(trace.controller.refine_passes ?? 0) > 0 ? (
              <span
                className="controller-detail-item"
                title="Critique → revise → re-score chain. Collapsed = revision rejected for falling back to the baseline answer."
              >
                Refine Chain:{" "}
                <strong>
                  {trace.controller.refine_accepted ?? 0}/{trace.controller.refine_total ?? 0} accepted
                </strong>
                {trace.controller.refine_collapsed
                  ? `, ${trace.controller.refine_collapsed} collapse-rejected`
                  : ""}
              </span>
            ) : (
              <span className="controller-detail-item">
                Branching: <strong>{trace.controller.branched ? "Yes" : "No"}</strong>
              </span>
            )}
            {trace.synthesisSources !== null && (
              <span className="controller-detail-item">
                Synthesis: <strong>Integrated {trace.synthesisSources} sources</strong>
              </span>
            )}
          </div>
        </div>
      )}

      {/* Quality-floor / collapse advisories */}
      {trace.selected !== null && !trace.floorMet && (
        <div className="trace-advisory warn">
          ⚠️ No candidate cleared the quality floor
          {trace.controller ? ` (${trace.controller.quality_floor.toFixed(2)})` : ""}.
          Showing the highest-quality option
          {trace.chosenQuality !== null ? ` (quality ${trace.chosenQuality.toFixed(2)})` : ""}
          {" "}— lower the floor or raise quality weight to accept more creative picks.
        </div>
      )}
      {trace.synthesisCollapsed && (
        <div className="trace-advisory warn">
          ⚠️ The merge collapsed onto the baseline answer and was rejected; the
          top-ranked candidate was used instead.
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
          <div className="section-title">
            Brainstormed Candidate Pool ({totalVariants})
            {trace.controller?.num_clusters !== undefined && (
              <span className="section-sub">
                {" "}· {trace.controller.num_clusters} semantic classes
              </span>
            )}
          </div>
          <div className="variant-grid">
            {trace.variants.map((item, i) => {
              const cids = trace.controller?.cluster_ids;
              const cid = cids ? cids[i] : undefined;
              const csize =
                cids && cid !== undefined && cid >= 0
                  ? cids.filter((c) => c === cid).length
                  : undefined;
              return (
                <VariantCard
                  key={i}
                  item={item}
                  score={trace.scores[i]}
                  onFrontier={trace.frontier[i] ?? false}
                  chosen={trace.selected === i}
                  synthesized={trace.synthesized}
                  clusterId={cid}
                  clusterSize={csize}
                  isHovered={hoveredIndex === i}
                  onMouseEnter={() => setHoveredIndex(i)}
                  onMouseLeave={() => setHoveredIndex(null)}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* Settings Used Card */}
      {configUsed && onApplyConfig && (
        <div className="settings-used-card card-container">
          <div className="settings-used-header">
            <span className="settings-used-title">
              ⚙️ Settings Used for this Turn
            </span>
            <button
              className="apply-config-btn"
              onClick={() => onApplyConfig(configUsed)}
              title="Apply these tuning parameters to composer"
            >
              Apply to Composer
            </button>
          </div>
          <div className="settings-grid">
            <div className="setting-stat">
              <span className="setting-stat-label">Profile</span>
              <span className="setting-stat-val">
                {(() => {
                  const pKey = getActiveProfileKey(configUsed);
                  return pKey === "mediumLow" ? "Medium Low" : pKey.charAt(0).toUpperCase() + pKey.slice(1);
                })()}
              </span>
            </div>
            <div className="setting-stat">
              <span className="setting-stat-label">k (Brainstorm)</span>
              <span className="setting-stat-val">{configUsed.k}</span>
            </div>
            <div className="setting-stat">
              <span className="setting-stat-label">Temp</span>
              <span className="setting-stat-val">{configUsed.temperature.toFixed(2)}</span>
            </div>
            <div className="setting-stat">
              <span className="setting-stat-label">Novelty Wt</span>
              <span className="setting-stat-val">{configUsed.novelty_weight.toFixed(2)}</span>
            </div>
            <div className="setting-stat">
              <span className="setting-stat-label">Coherence Wt</span>
              <span className="setting-stat-val">{configUsed.coherence_weight.toFixed(2)}</span>
            </div>
            <div className="setting-stat">
              <span className="setting-stat-label">Openness Wt</span>
              <span className="setting-stat-val">{configUsed.openness_weight.toFixed(2)}</span>
            </div>
            <div className="setting-stat">
              <span className="setting-stat-label">Originality Wt</span>
              <span className="setting-stat-val">{(configUsed.originality_weight ?? 0).toFixed(2)}</span>
            </div>
            <div className="setting-stat">
              <span className="setting-stat-label">Surprise Wt</span>
              <span className="setting-stat-val">{(configUsed.surprise_weight ?? 0).toFixed(2)}</span>
            </div>
            <div className="setting-stat">
              <span className="setting-stat-label">Quality Floor</span>
              <span className="setting-stat-val">{configUsed.convergent_floor.toFixed(2)}</span>
            </div>
            {configUsed.breadth_k > 0 && (
              <div className="setting-stat">
                <span className="setting-stat-label">Breadth k</span>
                <span className="setting-stat-val">{configUsed.breadth_k}</span>
              </div>
            )}
            {configUsed.refine_passes !== undefined && configUsed.refine_passes > 0 && (
              <div className="setting-stat">
                <span className="setting-stat-label">Refine Passes</span>
                <span className="setting-stat-val">{configUsed.refine_passes}</span>
              </div>
            )}
            {configUsed.trajectory && (
              <div className="setting-stat">
                <span className="setting-stat-label">Trajectory</span>
                <span className="setting-stat-val">Yes</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
