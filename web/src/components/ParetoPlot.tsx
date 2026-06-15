import type { Trace } from "../types";

interface ParetoPlotProps {
  trace: Trace;
  hoveredIndex: number | null;
  setHoveredIndex: (idx: number | null) => void;
  onSelectIndex?: (idx: number) => void;
}

export function ParetoPlot({
  trace,
  hoveredIndex,
  setHoveredIndex,
  onSelectIndex,
}: ParetoPlotProps) {
  const width = 340;
  const height = 240;
  const margin = { top: 25, right: 25, bottom: 40, left: 45 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const getX = (nov: number) => margin.left + nov * plotWidth;
  const getY = (qual: number) => margin.top + (1 - qual) * plotHeight;

  // Gather scored points
  const points = Object.entries(trace.scores).map(([idxStr, score]) => {
    const idx = Number(idxStr);
    const item = trace.variants[idx];
    const isModal = item?.is_modal ?? (idx === 0);
    const onFrontier = trace.frontier[idx] ?? false;
    const isChosen = trace.selected === idx;
    return {
      index: idx,
      novelty: score.novelty,
      quality: score.quality,
      isModal,
      onFrontier,
      isChosen,
      text: item?.text ?? "",
    };
  });

  // Calculate Pareto Frontier path
  // Only connect points that are scored AND marked as frontier
  const frontierPoints = points
    .filter((p) => p.onFrontier)
    .sort((a, b) => a.novelty - b.novelty);

  let frontierPathD = "";
  if (frontierPoints.length >= 2) {
    frontierPathD = frontierPoints
      .map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.novelty)} ${getY(p.quality)}`)
      .join(" ");
  }

  // Find active hovered point for tooltip
  const hoveredPoint = hoveredIndex !== null ? points.find((p) => p.index === hoveredIndex) : null;

  return (
    <div className="pareto-plot-container">
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${width} ${height}`}
        className="pareto-svg"
      >
        {/* Plot Background Grid */}
        {[0, 0.25, 0.5, 0.75, 1].map((val) => {
          const x = getX(val);
          const y = getY(val);
          return (
            <g key={val} className="grid-lines">
              {/* Vertical grid lines */}
              <line
                x1={x}
                y1={margin.top}
                x2={x}
                y2={height - margin.bottom}
                stroke="var(--line)"
                strokeDasharray="2 4"
                strokeWidth={0.5}
              />
              {/* Horizontal grid lines */}
              <line
                x1={margin.left}
                y1={y}
                x2={width - margin.right}
                y2={y}
                stroke="var(--line)"
                strokeDasharray="2 4"
                strokeWidth={0.5}
              />
              {/* X Axis labels */}
              <text
                x={x}
                y={height - margin.bottom + 16}
                textAnchor="middle"
                className="axis-label-tick"
              >
                {val.toFixed(2)}
              </text>
              {/* Y Axis labels */}
              <text
                x={margin.left - 8}
                y={y + 4}
                textAnchor="end"
                className="axis-label-tick"
              >
                {val.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* X Axis title */}
        <text
          x={margin.left + plotWidth / 2}
          y={height - 8}
          textAnchor="middle"
          className="axis-title"
        >
          Novelty (distance from modal) →
        </text>

        {/* Y Axis title */}
        <text
          transform={`rotate(-90, 12, ${margin.top + plotHeight / 2})`}
          x={12}
          y={margin.top + plotHeight / 2}
          textAnchor="middle"
          className="axis-title"
        >
          Quality Rubric Score ↑
        </text>

        {/* Pareto Frontier Line */}
        {frontierPathD && (
          <path
            d={frontierPathD}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            className="frontier-line"
          />
        )}

        {/* Plotted Points */}
        {points.map((p) => {
          const cx = getX(p.novelty);
          const cy = getY(p.quality);
          const isHovered = hoveredIndex === p.index;

          let r = 5;
          let fill = "var(--text-muted)";
          let stroke = "var(--line)";
          let className = "plot-dot";

          if (p.isChosen) {
            r = 7;
            fill = "var(--chosen)";
            stroke = "#0f1216";
            className += " chosen-dot";
          } else if (p.isModal) {
            r = 5.5;
            fill = "var(--panel2)";
            stroke = "var(--muted)";
            className += " modal-dot";
          } else if (p.onFrontier) {
            r = 5.5;
            fill = "var(--accent)";
            stroke = "#0f1216";
            className += " frontier-dot";
          } else {
            r = 5;
            fill = "var(--nov)";
            stroke = "#0f1216";
          }

          return (
            <g key={p.index}>
              {/* Pulsing ring for chosen dot */}
              {p.isChosen && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={12}
                  fill="none"
                  stroke="var(--chosen)"
                  strokeWidth={1.5}
                  opacity={0.4}
                  className="chosen-ring"
                />
              )}
              {/* Glow for hovered dot */}
              {isHovered && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={r + 4}
                  fill={fill}
                  opacity={0.25}
                />
              )}
              {/* Main Dot */}
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={fill}
                stroke={stroke}
                strokeWidth={isHovered ? 2 : 1.5}
                className={className}
                style={{ cursor: "pointer" }}
                onMouseEnter={() => setHoveredIndex(p.index)}
                onMouseLeave={() => setHoveredIndex(null)}
                onClick={() => onSelectIndex?.(p.index)}
              />
            </g>
          );
        })}

        {/* Tooltip Overlay */}
        {hoveredPoint && (
          <g>
            {/* Determine tooltip position to avoid clipping */}
            {(() => {
              const cx = getX(hoveredPoint.novelty);
              const cy = getY(hoveredPoint.quality);
              const tWidth = 140;
              const tHeight = 62;
              const xPos = cx > width - tWidth - 20 ? cx - tWidth - 10 : cx + 10;
              const yPos = cy > height - tHeight - 30 ? cy - tHeight - 10 : cy + 10;

              return (
                <g className="plot-tooltip">
                  <rect
                    x={xPos}
                    y={yPos}
                    width={tWidth}
                    height={tHeight}
                    rx={6}
                    fill="var(--panel)"
                    stroke="var(--line)"
                    strokeWidth={1}
                    className="tooltip-bg"
                  />
                  <text
                    x={xPos + 8}
                    y={yPos + 16}
                    className="tooltip-title"
                    fontWeight="600"
                  >
                    {hoveredPoint.isChosen
                      ? "✨ Chosen Reply"
                      : hoveredPoint.isModal
                      ? "🤖 Modal (Greedy)"
                      : hoveredPoint.onFrontier
                      ? "⚡ Frontier Candidate"
                      : `Variant #${hoveredPoint.index}`}
                  </text>
                  <text x={xPos + 8} y={yPos + 34} className="tooltip-stat">
                    Novelty:{" "}
                    <tspan fontWeight="500" fill="var(--nov)">
                      {hoveredPoint.novelty.toFixed(2)}
                    </tspan>
                  </text>
                  <text x={xPos + 8} y={yPos + 50} className="tooltip-stat">
                    Quality:{" "}
                    <tspan fontWeight="500" fill="var(--qual)">
                      {hoveredPoint.quality.toFixed(2)}
                    </tspan>
                  </text>
                </g>
              );
            })()}
          </g>
        )}
      </svg>
    </div>
  );
}
