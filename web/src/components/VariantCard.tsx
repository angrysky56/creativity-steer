import type { Scored, VariantItem } from "../types";

interface Props {
  item: VariantItem;
  score?: Scored;
  onFrontier: boolean;
  chosen: boolean;
  isHovered?: boolean;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}

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
  isHovered,
  onMouseEnter,
  onMouseLeave,
}: Props) {
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
          {onFrontier && <span className="tag frontier-tag">frontier</span>}
          {chosen && <span className="tag chosen-tag">chosen response</span>}
        </div>
      </div>
      <div className="variant-text">{item.text}</div>
      {score ? (
        <div className="variant-bars">
          {score.scores ? (
            Object.entries(score.scores).map(([name, val]) => {
              const kind = name === "novelty" ? "nov" : name === "quality" ? "qual" : name === "coherence" ? "coh" : name;
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
