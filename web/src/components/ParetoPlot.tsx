import type { Trace } from "../types";

interface ParetoPlotProps {
  trace: Trace;
  hoveredIndex: number | null;
  setHoveredIndex: (idx: number | null) => void;
  onSelectIndex?: (idx: number) => void;
}

// Wrap text to up to `maxLines` lines of ~`perLine` chars, ellipsizing overflow.
function wrap(text: string, perLine: number, maxLines: number): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let cur = "";
  for (const w of words) {
    if (cur && (cur + " " + w).length > perLine) {
      lines.push(cur);
      cur = w;
      if (lines.length >= maxLines) break;
    } else {
      cur = cur ? cur + " " + w : w;
    }
  }
  if (cur && lines.length < maxLines) lines.push(cur);
  const out = lines.slice(0, maxLines);
  if (out.length === maxLines && text.length > out.join(" ").length) {
    out[maxLines - 1] = out[maxLines - 1].replace(/.{1}$/, "…");
  }
  return out;
}

export function ParetoPlot({
  trace,
  hoveredIndex,
  setHoveredIndex,
  onSelectIndex,
}: ParetoPlotProps) {
  const width = 340;
  const height = 240;
  const margin = { top: 22, right: 22, bottom: 38, left: 45 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const getX = (nov: number) => margin.left + nov * plotWidth;
  const getY = (qual: number) => margin.top + (1 - qual) * plotHeight;
  // coherence (basin depth) -> dot radius
  const getR = (coh: number | undefined) => 4 + (coh ?? 0.5) * 5;

  const points = Object.entries(trace.scores).map(([idxStr, score]) => {
    const idx = Number(idxStr);
    const item = trace.variants[idx];
    return {
      index: idx,
      novelty: score.novelty,
      quality: score.quality,
      coherence: score.scores?.coherence,
      openness: score.scores?.openness,
      isModal: item?.is_modal ?? idx === 0,
      onFrontier: trace.frontier[idx] ?? false,
      isChosen: trace.selected === idx,
      text: item?.text ?? "",
    };
  });

  // Frontier path connects only the frontier points, left-to-right.
  const frontierPoints = points
    .filter((p) => p.onFrontier)
    .sort((a, b) => a.novelty - b.novelty);
  const frontierPathD =
    frontierPoints.length >= 2
      ? frontierPoints
          .map(
            (p, i) => `${i === 0 ? "M" : "L"} ${getX(p.novelty)} ${getY(p.quality)}`,
          )
          .join(" ")
      : "";

  const hoveredPoint =
    hoveredIndex !== null ? points.find((p) => p.index === hoveredIndex) : null;

  return (
    <div className="pareto-plot-container">
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${width} ${height}`}
        className="pareto-svg"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((val) => {
          const x = getX(val);
          const y = getY(val);
          return (
            <g key={val} className="grid-lines">
              <line x1={x} y1={margin.top} x2={x} y2={height - margin.bottom}
                stroke="var(--line)" strokeDasharray="2 4" strokeWidth={0.5} />
              <line x1={margin.left} y1={y} x2={width - margin.right} y2={y}
                stroke="var(--line)" strokeDasharray="2 4" strokeWidth={0.5} />
              <text x={x} y={height - margin.bottom + 14} textAnchor="middle"
                className="axis-label-tick">{val.toFixed(2)}</text>
              <text x={margin.left - 8} y={y + 4} textAnchor="end"
                className="axis-label-tick">{val.toFixed(2)}</text>
            </g>
          );
        })}

        <text x={margin.left + plotWidth / 2} y={height - 6} textAnchor="middle"
          className="axis-title">Novelty (distance from modal) →</text>

        {/* Y title: "→" renders as an UP arrow once the text is rotated -90°. */}
        <text
          transform={`rotate(-90, 12, ${margin.top + plotHeight / 2})`}
          x={12} y={margin.top + plotHeight / 2} textAnchor="middle"
          className="axis-title">Quality Rubric Score →</text>

        {frontierPathD && (
          <path d={frontierPathD} fill="none" stroke="var(--accent)"
            strokeWidth={1.5} strokeDasharray="4 3" className="frontier-line" />
        )}

        {points.map((p) => {
          const cx = getX(p.novelty);
          const cy = getY(p.quality);
          const isHovered = hoveredIndex === p.index;
          const r = getR(p.coherence) + (p.isChosen ? 1 : 0);

          let fill = "var(--nov)";
          let stroke = "#0f1216";
          if (p.isChosen) fill = "var(--chosen)";
          else if (p.isModal) {
            fill = "var(--panel2)";
            stroke = "var(--muted)";
          } else if (p.onFrontier) fill = "var(--accent)";

          const labelFill = p.isModal ? "var(--text)" : "#0f1216";

          return (
            <g key={p.index}
              style={{ cursor: "pointer" }}
              onMouseEnter={() => setHoveredIndex(p.index)}
              onMouseLeave={() => setHoveredIndex(null)}
              onClick={() => onSelectIndex?.(p.index)}>
              {/* openness (branching) -> outer ring */}
              {p.openness !== undefined && (
                <circle cx={cx} cy={cy} r={r + 3 + p.openness * 4} fill="none"
                  stroke="var(--opn)" strokeWidth={1.3}
                  opacity={0.2 + p.openness * 0.45} />
              )}
              {p.isChosen && (
                <circle cx={cx} cy={cy} r={r + 9} fill="none" stroke="var(--chosen)"
                  strokeWidth={1.5} opacity={0.4} className="chosen-ring" />
              )}
              {isHovered && (
                <circle cx={cx} cy={cy} r={r + 4} fill={fill} opacity={0.25} />
              )}
              <circle cx={cx} cy={cy} r={r} fill={fill} stroke={stroke}
                strokeWidth={isHovered ? 2 : 1.5}
                className={"plot-dot" + (p.isChosen ? " chosen-dot" : "")} />
              <text x={cx} y={cy + 2.6} textAnchor="middle" fontSize={7}
                fontWeight={600} fill={labelFill} pointerEvents="none">
                {p.isModal ? "M" : p.index}
              </text>
            </g>
          );
        })}

        {hoveredPoint && (() => {
          const cx = getX(hoveredPoint.novelty);
          const cy = getY(hoveredPoint.quality);
          const lines = wrap(hoveredPoint.text, 30, 2);
          const rows: Array<[string, string, string]> = [
            ["Novelty", hoveredPoint.novelty.toFixed(2), "var(--nov)"],
            ["Quality", hoveredPoint.quality.toFixed(2), "var(--qual)"],
          ];
          if (hoveredPoint.coherence !== undefined)
            rows.push(["Coherence", hoveredPoint.coherence.toFixed(2), "var(--coh)"]);
          if (hoveredPoint.openness !== undefined)
            rows.push(["Openness", hoveredPoint.openness.toFixed(2), "var(--opn)"]);

          const tWidth = 184;
          const tHeight = 22 + lines.length * 12 + rows.length * 15 + 8;
          const xPos = cx > width - tWidth - 14 ? cx - tWidth - 10 : cx + 12;
          const yPos = cy > height - tHeight - 16 ? cy - tHeight - 8 : cy + 8;
          const title = hoveredPoint.isChosen
            ? "✨ Chosen Reply"
            : hoveredPoint.isModal
              ? "🤖 Modal (Greedy)"
              : hoveredPoint.onFrontier
                ? "⚡ Frontier Candidate"
                : `Variant #${hoveredPoint.index}`;

          return (
            <g className="plot-tooltip">
              <rect x={xPos} y={yPos} width={tWidth} height={tHeight} rx={6}
                fill="var(--panel)" stroke="var(--line)" strokeWidth={1}
                className="tooltip-bg" />
              <text x={xPos + 8} y={yPos + 15} className="tooltip-title"
                fontWeight="600">{title}</text>
              {lines.map((ln, i) => (
                <text key={i} x={xPos + 8} y={yPos + 28 + i * 12}
                  className="tooltip-snippet" fill="var(--muted)" fontSize={9}>
                  {ln}
                </text>
              ))}
              {rows.map((row, i) => (
                <text key={row[0]} x={xPos + 8}
                  y={yPos + 28 + lines.length * 12 + i * 15} className="tooltip-stat">
                  {row[0]}:{" "}
                  <tspan fontWeight="500" fill={row[2]}>{row[1]}</tspan>
                </text>
              ))}
            </g>
          );
        })()}
      </svg>

      <div className="pareto-legend">
        <span><b>X</b> novelty · <b>Y</b> quality</span>
        <span><i className="lg-size" /> size = coherence</span>
        <span><i className="lg-ring" /> ring = openness</span>
        <span><i className="lg-dot chosen" /> chosen</span>
        <span><i className="lg-dot frontier" /> frontier</span>
        <span><i className="lg-dot modal" /> modal</span>
      </div>
    </div>
  );
}
