import { useRef, useState, useEffect } from "react";
import type { ChangeEvent } from "react";
import { streamChat } from "./api";
import { TracePanel } from "./components/Trace";
import { emptyTrace } from "./types";
import type { ChatMessage, Config, Trace } from "./types";

const DEFAULT_CONFIG: Config = {
  k: 5,
  novelty_weight: 0.5,
  convergent_floor: 0.34,
  temperature: 0.9,
};

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [config, setConfig] = useState<Config>(DEFAULT_CONFIG);
  const [trace, setTrace] = useState<Trace>(emptyTrace());
  const [streaming, setStreaming] = useState(false);
  const [activeMessageIndex, setActiveMessageIndex] = useState<number | null>(null);
  const [configExpanded, setConfigExpanded] = useState(false);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Check backend health on mount
  useEffect(() => {
    fetch("/api/health")
      .then((r) => {
        if (r.ok) {
          setHealthy(true);
        } else {
          setHealthy(false);
        }
      })
      .catch(() => setHealthy(false));
  }, []);

  // Scroll to bottom when messages list changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  async function send() {
    const message = input.trim();
    if (!message || streaming) return;
    const history = messages;
    setMessages([...history, { role: "user", content: message }]);
    setInput("");
    setTrace(emptyTrace());
    setActiveMessageIndex(null);
    setStreaming(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    let currentTrace = emptyTrace();

    try {
      for await (const ev of streamChat(message, history, config, ctrl.signal)) {
        currentTrace = reduce(currentTrace, ev);
        setTrace(currentTrace);

        if (ev.type === "response") {
          const newAssistantIndex = history.length + 1;
          setMessages((m) => [
            ...m,
            {
              role: "assistant",
              content: ev.text,
              trace: currentTrace,
            },
          ]);
          setActiveMessageIndex(newAssistantIndex);
        } else if (ev.type === "error") {
          setMessages((m) => [
            ...m,
            { role: "assistant", content: `⚠️ ${ev.message}` },
          ]);
        }
      }
    } catch (err) {
      if (ctrl.signal.aborted) {
        setMessages((m) => [
          ...m,
          { role: "assistant", content: "⚠️ Request cancelled by user." },
        ]);
      } else {
        setMessages((m) => [
          ...m,
          { role: "assistant", content: `⚠️ error: ${String(err)}` },
        ]);
      }
    } finally {
      setStreaming(false);
    }
  }

  function cancel() {
    if (abortRef.current) {
      abortRef.current.abort();
    }
  }

  function clearChat() {
    if (streaming) return;
    setMessages([]);
    setTrace(emptyTrace());
    setActiveMessageIndex(null);
  }

  const displayedTrace = streaming
    ? trace
    : activeMessageIndex !== null && messages[activeMessageIndex]?.trace
    ? (messages[activeMessageIndex].trace as Trace)
    : trace;

  return (
    <div className="app">
      <header className="navbar">
        <div className="logo-group">
          <h1>creativity-steer</h1>
          <span className="subtitle">brainstorm → score novelty + quality → select</span>
        </div>
        <div className="header-actions">
          {healthy === true && (
            <span className="health-badge healthy">
              <span className="indicator-dot"></span>
              Backend Online
            </span>
          )}
          {healthy === false && (
            <span className="health-badge unhealthy">
              <span className="indicator-dot"></span>
              Backend Offline
            </span>
          )}
          {messages.length > 0 && (
            <button className="clear-btn" onClick={clearChat} disabled={streaming}>
              Clear Chat
            </button>
          )}
        </div>
      </header>

      <div className="layout">
        <section className="chat-col">
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="welcome-hero">
                <div className="hero-logo">🧠✨</div>
                <h2>Creative Steer Assistant</h2>
                <p>
                  This assistant steers responses toward high creativity using Pareto optimization. It scores candidates based on their semantic novelty and quality.
                </p>
                <div className="quick-suggestions">
                  <p className="suggestion-title">Try asking things that benefit from unique angles:</p>
                  <button className="suggestion-card" onClick={() => setInput("What is a unique metaphor for the passage of time?")}>
                    "What is a unique metaphor for the passage of time?"
                  </button>
                  <button className="suggestion-card" onClick={() => setInput("Suggest a plot twist for a story about a library that exists outside of space.")}>
                    "Suggest a plot twist for a story about a library that exists outside of space."
                  </button>
                </div>
              </div>
            )}

            {messages.map((m, i) => {
              const isAssistant = m.role === "assistant";
              const hasTrace = isAssistant && !!m.trace;
              const isSelected = activeMessageIndex === i;

              return (
                <div
                  key={i}
                  className={`msg-wrapper ${m.role} ${isSelected ? "selected-turn" : ""}`}
                >
                  <div className={`msg ${m.role}`}>
                    <div
                      className={`bubble ${hasTrace ? "steer-bubble" : ""}`}
                      onClick={() => isAssistant && hasTrace && setActiveMessageIndex(i)}
                      title={hasTrace ? "Click to view decision trace analytics" : undefined}
                      style={{ cursor: hasTrace ? "pointer" : "default" }}
                    >
                      <div className="bubble-content">{m.content}</div>
                      {hasTrace && (
                        <div className="steer-meta">
                          <span className="steer-pill">
                            ✨ Steered (k={m.trace?.variants?.length ?? config.k})
                          </span>
                          <span className="click-hint">Inspect Trace</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
            {streaming && (
              <div className="msg-wrapper assistant">
                <div className="msg assistant">
                  <div className="bubble thinking-bubble">
                    <div className="thinking-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    <div className="thinking-text">Steering candidate responses...</div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="composer-area">
            {/* Collapsible Tuning Panel */}
            <div className={`config-drawer ${configExpanded ? "expanded" : ""}`}>
              <button
                className="drawer-toggle"
                onClick={() => setConfigExpanded(!configExpanded)}
              >
                <span className="toggle-icon">⚙️</span>
                <span className="toggle-label">Tuning Parameters</span>
                <span className="toggle-summary">
                  {!configExpanded &&
                    `k=${config.k} | novelty=${config.novelty_weight.toFixed(2)} | floor=${config.convergent_floor.toFixed(2)} | temp=${config.temperature.toFixed(2)}`}
                </span>
                <span className="chevron">{configExpanded ? "▼" : "▲"}</span>
              </button>
              {configExpanded && (
                <Controls config={config} setConfig={setConfig} disabled={streaming} />
              )}
            </div>

            <div className="composer">
              <textarea
                value={input}
                placeholder={streaming ? "Waiting for response..." : "Ask something creative..."}
                onChange={(e) => setInput(e.target.value)}
                disabled={streaming}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void send();
                  }
                }}
              />
              {streaming ? (
                <button className="cancel-btn" onClick={cancel}>
                  Stop
                </button>
              ) : (
                <button
                  className="send-btn"
                  onClick={() => void send()}
                  disabled={!input.trim()}
                >
                  Send
                </button>
              )}
            </div>
          </div>
        </section>

        <section className="trace-col">
          <TracePanel trace={displayedTrace} streaming={streaming} />
        </section>
      </div>
    </div>
  );
}

function reduce(t: Trace, ev: import("./types").TraceEvent): Trace {
  switch (ev.type) {
    case "modal":
      return { ...t, modal: ev.text };
    case "variants":
      return { ...t, variants: ev.items };
    case "scored":
      return { ...t, scores: { ...t.scores, [ev.index]: ev } };
    case "selected":
      return { ...t, frontier: ev.frontier, selected: ev.index };
    case "response":
      return { ...t, done: true };
    default:
      return t;
  }
}

interface CtlProps {
  config: Config;
  setConfig: (c: Config) => void;
  disabled: boolean;
}

function Controls({ config, setConfig, disabled }: CtlProps) {
  const set = (key: keyof Config) => (e: ChangeEvent<HTMLInputElement>) =>
    setConfig({ ...config, [key]: Number(e.target.value) });

  return (
    <div className="controls-grid">
      <div className="control-item">
        <div className="control-header">
          <label htmlFor="variants-k">Brainstorm Size (k)</label>
          <span className="control-val">{config.k}</span>
        </div>
        <input
          id="variants-k"
          type="range"
          min={3}
          max={8}
          step={1}
          value={config.k}
          onChange={set("k")}
          disabled={disabled}
        />
        <div className="control-desc">Number of unique candidates to generate.</div>
      </div>

      <div className="control-item">
        <div className="control-header">
          <label htmlFor="novelty-weight">Novelty Weight</label>
          <span className="control-val">{config.novelty_weight.toFixed(2)}</span>
        </div>
        <input
          id="novelty-weight"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={config.novelty_weight}
          onChange={set("novelty_weight")}
          disabled={disabled}
        />
        <div className="control-desc">Preference for novelty vs. quality score.</div>
      </div>

      <div className="control-item">
        <div className="control-header">
          <label htmlFor="quality-floor">Quality Floor</label>
          <span className="control-val">{config.convergent_floor.toFixed(2)}</span>
        </div>
        <input
          id="quality-floor"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={config.convergent_floor}
          onChange={set("convergent_floor")}
          disabled={disabled}
        />
        <div className="control-desc">Minimum rubric grade required to select.</div>
      </div>

      <div className="control-item">
        <div className="control-header">
          <label htmlFor="temperature">Creativity (Temp)</label>
          <span className="control-val">{config.temperature.toFixed(2)}</span>
        </div>
        <input
          id="temperature"
          type="range"
          min={0}
          max={1.5}
          step={0.05}
          value={config.temperature}
          onChange={set("temperature")}
          disabled={disabled}
        />
        <div className="control-desc">Randomness of alternative brainstorming.</div>
      </div>
    </div>
  );
}
