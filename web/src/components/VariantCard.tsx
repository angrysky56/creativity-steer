import type { Scored, VariantItem } from "../types";

interface Props {
  item: VariantItem;
  score?: Scored;
  onFrontier: boolean;
  chosen: boolean;
}

function Bar({ label, value, kind }: { label: string; value: number; kind: string }) {
  return (
    <div className="bar-row">
      <span className="bar-label">{label}</span>
      <div className="bar-track">
        <div className={`bar-fill ${kind}`} style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
      <span className="bar-val">{value.toFixed(2)}</span>
    </div>
  );
}

export function VariantCard({ item, score, onFrontier, chosen }: Props) {
  return (
    <div className={`variant-card${chosen ? " chosen" : ""}${onFrontier ? " frontier" : ""}`}>
      <div className="variant-head">
        {item.is_modal && <span className="tag modal">modal</span>}
        {onFrontier && <span className="tag frontier-tag">frontier</span>}
        {chosen && <span className="tag chosen-tag">chosen</span>}
      </div>
      <div className="variant-text">{item.text}</div>
      {score ? (
        <div className="variant-bars">
          <Bar label="novelty" value={score.novelty} kind="nov" />
          <Bar label="quality" value={score.quality} kind="qual" />
        </div>
      ) : (
        <div className="variant-pending">scoring…</div>
      )}
    </div>
  );
}
