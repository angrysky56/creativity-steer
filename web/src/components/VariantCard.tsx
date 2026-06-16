import type { Scored, VariantItem } from "../types";

interface Props {
  item: VariantItem;
  score?: Scored;
  onFrontier: boolean;
  chosen: boolean;
  synthesized?: boolean;
  clusterId?: number;
  clusterSize?: number;
  isHovered?: boolean;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}

// Stable palette for semantic-class chips (candidates in the same class are
// paraphrases of one idea per bidirectional entailment).
const CLUSTER_HUES = [200, 280, 30, 130, 340, 90, 250, 10, 170, 310];

function Bar({
  label,
  value,
  kind,
}: {
  label: string;
  value: number;
  kind: string;
}) {
  return (
    <div className="bar-row">
      <span className="bar-label">{label}</span>
      <div className="bar-track">
        <div
          className={`bar-fill ${kind}`}
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
      <span className="bar-val">{value.toFixed(2)}</span>
    </div>
  );
}

export function VariantCard({
  item,
  score,
  onFrontier,
  chosen,
  synthesized,
  clusterId,
  clusterSize,
  isHovered,
  onMouseEnter,
  onMouseLeave,
}: Props) {
  const hasCluster = clusterId !== undefined && clusterId >= 0;
  const hue = hasCluster ? CLUSTER_HUES[clusterId % CLUSTER_HUES.length] : 0;
  return (
    <div
      className={`variant-card${chosen ? " chosen" : ""}${
        onFrontier ? " frontier" : ""
      }${isHovered ? " card-hovered" : ""}`}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="variant-head">
        <div className="variant-badges">
          {item.is_modal && <span className="tag modal">modal (greedy)</span>}
          {hasCluster && (
            <span
              className="tag cluster-tag"
              title={
                clusterSize && clusterSize > 1
                  ? `Semantic class ${clusterId} — ${clusterSize} candidates share this meaning`
                  : `Semantic class ${clusterId} — unique meaning`
              }
              style={{
                background: `hsl(${hue} 70% 18%)`,
                color: `hsl(${hue} 80% 75%)`,
                borderColor: `hsl(${hue} 60% 35%)`,
              }}
            >
              class {clusterId}
              {clusterSize && clusterSize > 1 ? ` ×${clusterSize}` : ""}
            </span>
          )}
          {onFrontier && <span className="tag frontier-tag">frontier</span>}
          {chosen && (
            <span className="tag chosen-tag">
              {synthesized ? "top ranked (merged)" : "chosen response"}
            </span>
          )}
        </div>
      </div>
      <div className="variant-text">{item.text}</div>
      {score ? (
        <div className="variant-bars">
          {score.scores ? (
            Object.entries(score.scores).map(([name, val]) => {
              const kind =
                name === "novelty"
                  ? "nov"
                  : name === "quality"
                    ? "qual"
                    : name === "coherence"
                      ? "coh"
                      : name === "openness"
                        ? "opn"
                        : name === "originality"
                          ? "orig"
                          : name === "surprise"
                            ? "surp"
                            : name;
              return <Bar key={name} label={name} value={val} kind={kind} />;
            })
          ) : (
            <>
              <Bar label="novelty" value={score.novelty} kind="nov" />
              <Bar label="quality" value={score.quality} kind="qual" />
            </>
          )}
        </div>
      ) : (
        <div className="variant-pending">
          <span className="loading-dots">scoring</span>
        </div>
      )}
    </div>
  );
}
