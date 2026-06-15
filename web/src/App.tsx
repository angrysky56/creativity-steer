import { useRef, useState, useEffect } from "react";
import type { ChangeEvent } from "react";
import { streamChat } from "./api";
import { TracePanel } from "./components/Trace";
import { emptyTrace } from "./types";
import type { ChatMessage, Config, Trace } from "./types";

const PROFILES = {
  low: {
    name: "Low Exploration (Deterministic)",
    desc: "Focuses on high-quality, closely aligned answers. Fast and highly coherent.",
    config: {
      k: 3,
      novelty_weight: 0.15,
      coherence_weight: 0.3,
      openness_weight: 0.0,
      convergent_floor: 0.55,
      temperature: 0.4,
      openness_branches: 0,
      breadth_k: 0,
      prime_n: 0,
      branch: false,
      synthesize: false,
    },
  },
  mediumLow: {
    name: "Medium Low (Balanced & Grounded)",
    desc: "Balanced, moderately safe exploration with guided candidate selection.",
    config: {
      k: 5,
      novelty_weight: 0.35,
      coherence_weight: 0.2,
      openness_weight: 0.0,
      convergent_floor: 0.4,
      temperature: 0.7,
      openness_branches: 0,
      breadth_k: 10,
      prime_n: 4,
      branch: false,
      synthesize: false,
    },
  },
  medium: {
    name: "Medium (Creative)",
    desc: "The default creative steer profile. Explores unique alternative perspectives.",
    config: {
      k: 5,
      novelty_weight: 0.5,
      coherence_weight: 0.0,
      openness_weight: 0.0,
      convergent_floor: 0.34,
      temperature: 0.9,
      openness_branches: 0,
      breadth_k: 0,
      prime_n: 0,
      branch: false,
      synthesize: false,
    },
  },
  high: {
    name: "High (Wild Breadth & Deep Synthesis)",
    desc: "Wide candidate pool funneled, analyzed for counterfactual openness, deepened, and merged.",
    config: {
      k: 8,
      novelty_weight: 0.5,
      coherence_weight: 0.15,
      openness_weight: 0.25,
      convergent_floor: 0.3,
      temperature: 1.2,
      openness_branches: 3,
      breadth_k: 15,
      prime_n: 5,
      branch: true,
      synthesize: true,
    },
  },
};

const getActiveProfileKey = (currentConfig: Config): string => {
  for (const [key, profile] of Object.entries(PROFILES)) {
    const pc = profile.config;
    const match =
      currentConfig.k === pc.k &&
      Math.abs(currentConfig.novelty_weight - pc.novelty_weight) < 0.01 &&
      Math.abs(currentConfig.coherence_weight - pc.coherence_weight) < 0.01 &&
      Math.abs(currentConfig.openness_weight - pc.openness_weight) < 0.01 &&
      Math.abs(currentConfig.convergent_floor - pc.convergent_floor) < 0.01 &&
      Math.abs(currentConfig.temperature - pc.temperature) < 0.01 &&
      currentConfig.breadth_k === pc.breadth_k &&
      currentConfig.prime_n === pc.prime_n &&
      currentConfig.branch === pc.branch &&
      currentConfig.synthesize === pc.synthesize &&
      currentConfig.openness_branches === pc.openness_branches;
    if (match) return key;
  }
  return "custom";
};

const DEFAULT_CONFIG: Config = PROFILES.mediumLow.config;

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
                  {!configExpanded && (() => {
                    const profileKey = getActiveProfileKey(config);
                    const profileLabel = profileKey === "mediumLow" ? "Medium Low" : profileKey.charAt(0).toUpperCase() + profileKey.slice(1);
                    return `Profile: ${profileLabel} | k=${config.k} | novelty=${config.novelty_weight.toFixed(2)} | temp=${config.temperature.toFixed(2)}`;
                  })()}
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
      return { ...t, done: true, synthesized: ev.synthesized ?? false };
    case "controller":
      return { ...t, controller: ev };
    case "synthesis":
      return { ...t, synthesisSources: ev.sources };
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
  const setNum = (key: keyof Config) => (e: ChangeEvent<HTMLInputElement>) =>
    setConfig({ ...config, [key]: Number(e.target.value) });
  const setBool = (key: keyof Config) => (e: ChangeEvent<HTMLInputElement>) =>
    setConfig({ ...config, [key]: e.target.checked });

  const activeProfile = getActiveProfileKey(config);

  return (
    <div className="controls-container">
      {/* Exploration Profiles Selector */}
      <div className="profile-selector-section">
        <div className="profile-pills">
          {Object.entries(PROFILES).map(([key, p]) => (
            <button
              key={key}
              type="button"
              className={`profile-pill ${activeProfile === key ? "active" : ""}`}
              onClick={() => setConfig(p.config)}
              disabled={disabled}
            >
              {key === "mediumLow" ? "Medium Low" : key.charAt(0).toUpperCase() + key.slice(1)}
            </button>
          ))}
          {activeProfile === "custom" && (
            <span className="profile-pill active custom">Custom</span>
          )}
        </div>
        <p className="profile-desc">
          {activeProfile !== "custom"
            ? PROFILES[activeProfile as keyof typeof PROFILES].desc
            : "Manual adjustment of individual creativity and funnel parameters."}
        </p>
      </div>

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
            onChange={setNum("k")}
            disabled={disabled}
          />
          <div className="control-desc">Number of unique candidates to generate.</div>
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
            onChange={setNum("temperature")}
            disabled={disabled}
          />
          <div className="control-desc">Randomness of alternative brainstorming.</div>
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
            onChange={setNum("novelty_weight")}
            disabled={disabled}
          />
          <div className="control-desc">Preference for novelty vs. quality score.</div>
        </div>

        <div className="control-item">
          <div className="control-header">
            <label htmlFor="coherence-weight">Coherence Weight</label>
            <span className="control-val">{config.coherence_weight.toFixed(2)}</span>
          </div>
          <input
            id="coherence-weight"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={config.coherence_weight}
            onChange={setNum("coherence_weight")}
            disabled={disabled}
          />
          <div className="control-desc">Preference for stable idea attractor basin depth.</div>
        </div>

        <div className="control-item">
          <div className="control-header">
            <label htmlFor="openness-weight">Openness Weight</label>
            <span className="control-val">{config.openness_weight.toFixed(2)}</span>
          </div>
          <input
            id="openness-weight"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={config.openness_weight}
            onChange={setNum("openness_weight")}
            disabled={disabled}
          />
          <div className="control-desc">Preference for counterfactual branching openness score.</div>
        </div>

        <div className="control-item">
          <div className="control-header">
            <label htmlFor="openness-branches">Openness Branches</label>
            <span className="control-val">{config.openness_branches === 0 ? "Off" : config.openness_branches}</span>
          </div>
          <input
            id="openness-branches"
            type="range"
            min={0}
            max={10}
            step={1}
            value={config.openness_branches}
            onChange={setNum("openness_branches")}
            disabled={disabled}
          />
          <div className="control-desc">Number of continuation branches to probe (0 to disable).</div>
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
            onChange={setNum("convergent_floor")}
            disabled={disabled}
          />
          <div className="control-desc">Minimum rubric grade required to select.</div>
        </div>

        <div className="control-item">
          <div className="control-header">
            <label htmlFor="breadth-k">Breadth Candidates (breadth_k)</label>
            <span className="control-val">{config.breadth_k === 0 ? "Off" : config.breadth_k}</span>
          </div>
          <input
            id="breadth-k"
            type="range"
            min={0}
            max={50}
            step={5}
            value={config.breadth_k}
            onChange={setNum("breadth_k")}
            disabled={disabled}
          />
          <div className="control-desc">Generate this many total candidates (0 to disable).</div>
        </div>

        <div className="control-item">
          <div className="control-header">
            <label htmlFor="prime-n">Funnel Primes (prime_n)</label>
            <span className="control-val">{config.prime_n === 0 ? "Off" : config.prime_n}</span>
          </div>
          <input
            id="prime-n"
            type="range"
            min={0}
            max={15}
            step={1}
            value={config.prime_n}
            onChange={setNum("prime_n")}
            disabled={disabled}
          />
          <div className="control-desc">Keep this many diverse primes from candidate pool (0 to disable).</div>
        </div>

        <div className="control-item toggles-item">
          <div className="toggles-row">
            <label className="toggle-container" htmlFor="branch-check">
              <input
                id="branch-check"
                type="checkbox"
                checked={config.branch}
                onChange={setBool("branch")}
                disabled={disabled}
              />
              <span className="toggle-label-text">Deepen Candidates (Branch)</span>
            </label>

            <label className="toggle-container" htmlFor="synthesize-check">
              <input
                id="synthesize-check"
                type="checkbox"
                checked={config.synthesize}
                onChange={setBool("synthesize")}
                disabled={disabled}
              />
              <span className="toggle-label-text">Synthesize Frontier (Merge)</span>
            </label>
          </div>
          <div className="control-desc" style={{ marginTop: "4px" }}>
            Branching deepens primes. Synthesis merges the Pareto frontier into the final response.
          </div>
        </div>
      </div>
    </div>
  );
}
