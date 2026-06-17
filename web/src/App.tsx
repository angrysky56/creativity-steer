import { useRef, useState, useEffect } from "react";
import type { ChangeEvent } from "react";
import { streamChat, flagCorrection } from "./api";
import { TracePanel } from "./components/Trace";
import { emptyTrace } from "./types";
import type { ChatMessage, Config, Trace, ChatSession } from "./types";

export const PROFILES = {
  low: {
    name: "Low Exploration (Deterministic)",
    desc: "Focuses on high-quality, closely aligned answers. Fast and highly coherent.",
    config: {
      k: 3,
      novelty_weight: 0.15,
      coherence_weight: 0.3,
      openness_weight: 0.0,
      originality_weight: 0.0,
      surprise_weight: 0.0,
      convergent_floor: 0.55,
      temperature: 0.4,
      openness_branches: 0,
      breadth_k: 0,
      prime_n: 0,
      branch: false,
      synthesize: false,
      trajectory: false,
      refine_passes: 0,
      response_tokens: 0,
      brainstorm_tokens: 700,
      modal_tokens: 256,
      branch_tokens: 0,
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
      originality_weight: 0.0,
      surprise_weight: 0.0,
      convergent_floor: 0.4,
      temperature: 0.7,
      openness_branches: 0,
      breadth_k: 10,
      prime_n: 4,
      branch: false,
      synthesize: false,
      trajectory: false,
      refine_passes: 0,
      response_tokens: 0,
      brainstorm_tokens: 700,
      modal_tokens: 256,
      branch_tokens: 0,
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
      originality_weight: 0.0,
      surprise_weight: 0.0,
      convergent_floor: 0.34,
      temperature: 0.9,
      openness_branches: 0,
      breadth_k: 0,
      prime_n: 0,
      branch: false,
      synthesize: false,
      trajectory: false,
      refine_passes: 0,
      response_tokens: 0,
      brainstorm_tokens: 700,
      modal_tokens: 256,
      branch_tokens: 0,
    },
  },
  high: {
    name: "High (Connected Chains & Deep Synthesis)",
    desc: "Cluster-aware trajectory breadth → funnel → guided refine chain (critique→revise→re-score, collapse-guarded) → integrative synthesis.",
    config: {
      k: 8,
      novelty_weight: 0.4,
      coherence_weight: 0.15,
      openness_weight: 0.2,
      originality_weight: 0.2,
      surprise_weight: 0.25,
      convergent_floor: 0.3,
      temperature: 1.2,
      openness_branches: 3,
      breadth_k: 15,
      prime_n: 5,
      branch: false,
      synthesize: true,
      trajectory: true,
      refine_passes: 2,
      response_tokens: 0,
      brainstorm_tokens: 700,
      modal_tokens: 256,
      branch_tokens: 0,
    },
  },
};

export const getActiveProfileKey = (currentConfig: Config): string => {
  for (const [key, profile] of Object.entries(PROFILES)) {
    const pc = profile.config;
    const match =
      currentConfig.k === pc.k &&
      Math.abs(currentConfig.novelty_weight - pc.novelty_weight) < 0.01 &&
      Math.abs(currentConfig.coherence_weight - pc.coherence_weight) < 0.01 &&
      Math.abs(currentConfig.openness_weight - pc.openness_weight) < 0.01 &&
      Math.abs(
        (currentConfig.originality_weight ?? 0) - (pc.originality_weight ?? 0),
      ) < 0.01 &&
      Math.abs(
        (currentConfig.surprise_weight ?? 0) - (pc.surprise_weight ?? 0),
      ) < 0.01 &&
      Math.abs(currentConfig.convergent_floor - pc.convergent_floor) < 0.01 &&
      Math.abs(currentConfig.temperature - pc.temperature) < 0.01 &&
      currentConfig.breadth_k === pc.breadth_k &&
      currentConfig.prime_n === pc.prime_n &&
      currentConfig.branch === pc.branch &&
      currentConfig.synthesize === pc.synthesize &&
      currentConfig.openness_branches === pc.openness_branches &&
      (currentConfig.trajectory ?? false) === (pc.trajectory ?? false) &&
      (currentConfig.refine_passes ?? 0) === (pc.refine_passes ?? 0) &&
      (currentConfig.response_tokens ?? 0) === (pc.response_tokens ?? 0) &&
      (currentConfig.brainstorm_tokens ?? 700) ===
        (pc.brainstorm_tokens ?? 700) &&
      (currentConfig.modal_tokens ?? 256) === (pc.modal_tokens ?? 256) &&
      (currentConfig.branch_tokens ?? 0) === (pc.branch_tokens ?? 0);
    if (match) return key;
  }
  return "custom";
};

const DEFAULT_CONFIG: Config = PROFILES.mediumLow.config;

export function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [config, setConfig] = useState<Config>(DEFAULT_CONFIG);
  const [trace, setTrace] = useState<Trace>(emptyTrace());
  const [streaming, setStreaming] = useState(false);
  const [activeMessageIndex, setActiveMessageIndex] = useState<number | null>(
    null,
  );
  const [configExpanded, setConfigExpanded] = useState(false);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [flagged, setFlagged] = useState<Record<number, boolean>>({});

  async function markWrong(i: number) {
    const failed = messages[i]?.content ?? "";
    const prior = messages[i - 1]?.role === "user" ? messages[i - 1].content : "";
    const ok = await flagCorrection(failed, prior);
    if (ok) setFlagged((f) => ({ ...f, [i]: true }));
  }

  const abortRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Check backend health on mount & Load sessions from LocalStorage
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

    const saved = localStorage.getItem("cs_sessions");
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as ChatSession[];
        if (parsed.length > 0) {
          setSessions(parsed);
          setCurrentSessionId(parsed[0].id);
          setMessages(parsed[0].messages);
          setConfig(parsed[0].config || DEFAULT_CONFIG);
          return;
        }
      } catch (e) {
        console.error("Failed to load sessions", e);
      }
    }

    // Fallback to fresh session if none exist
    const firstSession: ChatSession = {
      id: crypto.randomUUID
        ? crypto.randomUUID()
        : Math.random().toString(36).substring(2),
      title: "New Chat",
      messages: [],
      config: DEFAULT_CONFIG,
    };
    setSessions([firstSession]);
    setCurrentSessionId(firstSession.id);
    setMessages([]);
    setConfig(DEFAULT_CONFIG);
  }, []);

  // Scroll to bottom when messages list changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  const updateCurrentSession = (
    updatedMessages: ChatMessage[],
    updatedConfig?: Config,
  ) => {
    if (!currentSessionId) return;
    setSessions((prev) => {
      const updated = prev.map((s) => {
        if (s.id === currentSessionId) {
          const firstUserMessage = updatedMessages.find(
            (m) => m.role === "user",
          );
          let title = s.title;
          if ((s.title === "New Chat" || s.title === "") && firstUserMessage) {
            title =
              firstUserMessage.content.slice(0, 30) +
              (firstUserMessage.content.length > 30 ? "..." : "");
          }
          return {
            ...s,
            title,
            messages: updatedMessages,
            config: updatedConfig !== undefined ? updatedConfig : s.config,
          };
        }
        return s;
      });
      localStorage.setItem("cs_sessions", JSON.stringify(updated));
      return updated;
    });
  };

  const handleConfigChange = (newConfig: Config) => {
    setConfig(newConfig);
    updateCurrentSession(messages, newConfig);
  };

  const selectSession = (id: string) => {
    if (streaming) return;
    const session = sessions.find((s) => s.id === id);
    if (!session) return;
    setCurrentSessionId(id);
    setMessages(session.messages);
    setConfig(session.config || DEFAULT_CONFIG);
    setTrace(emptyTrace());
    setActiveMessageIndex(null);
  };

  const createSession = () => {
    if (streaming) return;
    const newSession: ChatSession = {
      id: crypto.randomUUID
        ? crypto.randomUUID()
        : Math.random().toString(36).substring(2),
      title: "New Chat",
      messages: [],
      config: DEFAULT_CONFIG,
    };
    const updated = [newSession, ...sessions];
    setSessions(updated);
    setCurrentSessionId(newSession.id);
    setMessages([]);
    setConfig(DEFAULT_CONFIG);
    setTrace(emptyTrace());
    setActiveMessageIndex(null);
    localStorage.setItem("cs_sessions", JSON.stringify(updated));
  };

  const deleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (streaming) return;
    const updated = sessions.filter((s) => s.id !== id);
    setSessions(updated);
    localStorage.setItem("cs_sessions", JSON.stringify(updated));
    if (currentSessionId === id) {
      if (updated.length > 0) {
        selectSession(updated[0].id);
      } else {
        const newSession: ChatSession = {
          id: crypto.randomUUID
            ? crypto.randomUUID()
            : Math.random().toString(36).substring(2),
          title: "New Chat",
          messages: [],
          config: DEFAULT_CONFIG,
        };
        setSessions([newSession]);
        setCurrentSessionId(newSession.id);
        setMessages([]);
        setConfig(DEFAULT_CONFIG);
        setTrace(emptyTrace());
        setActiveMessageIndex(null);
        localStorage.setItem("cs_sessions", JSON.stringify([newSession]));
      }
    }
  };

  const renameSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const session = sessions.find((s) => s.id === id);
    if (!session) return;
    const newTitle = prompt("Enter new title for this chat:", session.title);
    if (newTitle === null || !newTitle.trim()) return;
    setSessions((prev) => {
      const updated = prev.map((s) =>
        s.id === id ? { ...s, title: newTitle.trim() } : s,
      );
      localStorage.setItem("cs_sessions", JSON.stringify(updated));
      return updated;
    });
  };

  async function send() {
    const message = input.trim();
    if (!message || streaming) return;
    const history = [...messages];
    const userMsg: ChatMessage = { role: "user", content: message };
    const updatedWithUser = [...history, userMsg];
    setMessages(updatedWithUser);
    updateCurrentSession(updatedWithUser);
    setInput("");
    setTrace(emptyTrace());
    setActiveMessageIndex(null);
    setStreaming(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    let currentTrace = emptyTrace();

    try {
      for await (const ev of streamChat(
        message,
        history,
        config,
        ctrl.signal,
      )) {
        currentTrace = reduce(currentTrace, ev);
        setTrace(currentTrace);

        if (ev.type === "response") {
          const assistantMsg: ChatMessage = {
            role: "assistant",
            content: ev.text,
            trace: currentTrace,
            config: { ...config },
          };
          const finalMessages = [...updatedWithUser, assistantMsg];
          setMessages(finalMessages);
          updateCurrentSession(finalMessages);
          setActiveMessageIndex(finalMessages.length - 1);
        } else if (ev.type === "error") {
          const errorMsg: ChatMessage = {
            role: "assistant",
            content: `⚠️ ${ev.message}`,
          };
          const finalMessages = [...updatedWithUser, errorMsg];
          setMessages(finalMessages);
          updateCurrentSession(finalMessages);
        }
      }
    } catch (err) {
      if (ctrl.signal.aborted) {
        const cancelMsg: ChatMessage = {
          role: "assistant",
          content: "⚠️ Request cancelled by user.",
        };
        const finalMessages = [...updatedWithUser, cancelMsg];
        setMessages(finalMessages);
        updateCurrentSession(finalMessages);
      } else {
        const errorMsg: ChatMessage = {
          role: "assistant",
          content: `⚠️ error: ${String(err)}`,
        };
        const finalMessages = [...updatedWithUser, errorMsg];
        setMessages(finalMessages);
        updateCurrentSession(finalMessages);
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
    updateCurrentSession([]);
  }

  const displayedTrace = streaming
    ? trace
    : activeMessageIndex !== null && messages[activeMessageIndex]?.trace
      ? (messages[activeMessageIndex].trace as Trace)
      : trace;

  const displayedConfig = streaming
    ? null
    : activeMessageIndex !== null && messages[activeMessageIndex]?.config
      ? (messages[activeMessageIndex].config as Config)
      : null;

  return (
    <div className="app">
      <header className="navbar">
        <div className="logo-group">
          <button
            className="sidebar-toggle-btn"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            title="Toggle Sidebar"
          >
            ☰
          </button>
          <h1>creativity-steer</h1>
          <span className="subtitle">
            brainstorm → score novelty + quality → select
          </span>
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
            <button
              className="clear-btn"
              onClick={clearChat}
              disabled={streaming}
            >
              Clear Chat
            </button>
          )}
        </div>
      </header>

      <div className="layout">
        {sidebarOpen && (
          <aside className="sidebar">
            <button
              className="new-chat-btn"
              onClick={createSession}
              disabled={streaming}
            >
              + New Chat
            </button>
            <div className="sessions-list">
              {sessions.map((s) => (
                <div
                  key={s.id}
                  className={`session-item ${s.id === currentSessionId ? "active" : ""}`}
                  onClick={() => selectSession(s.id)}
                >
                  <span className="session-title" title={s.title}>
                    💬 {s.title}
                  </span>
                  <div className="session-actions">
                    <button
                      className="session-action-btn rename"
                      onClick={(e) => renameSession(s.id, e)}
                      title="Rename Chat"
                    >
                      ✏️
                    </button>
                    <button
                      className="session-action-btn delete"
                      onClick={(e) => deleteSession(s.id, e)}
                      title="Delete Chat"
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </aside>
        )}

        <section className="chat-col">
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="welcome-hero">
                <div className="hero-logo">🧠✨</div>
                <h2>Creative Steer Assistant</h2>
                <p>
                  This assistant steers responses toward high creativity using
                  Pareto optimization. It scores candidates based on their
                  semantic novelty and quality.
                </p>
                <div className="quick-suggestions">
                  <p className="suggestion-title">
                    Try asking things that benefit from unique angles:
                  </p>
                  <button
                    className="suggestion-card"
                    onClick={() =>
                      setInput(
                        "What is a unique metaphor for the passage of time?",
                      )
                    }
                  >
                    "What is a unique metaphor for the passage of time?"
                  </button>
                  <button
                    className="suggestion-card"
                    onClick={() =>
                      setInput(
                        "Suggest a plot twist for a story about a library that exists outside of space.",
                      )
                    }
                  >
                    "Suggest a plot twist for a story about a library that
                    exists outside of space."
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
                      onClick={() =>
                        isAssistant && hasTrace && setActiveMessageIndex(i)
                      }
                      title={
                        hasTrace
                          ? "Click to view decision trace analytics"
                          : undefined
                      }
                      style={{ cursor: hasTrace ? "pointer" : "default" }}
                    >
                      <div className="bubble-content">{m.content}</div>
                      {hasTrace && (
                        <div className="steer-meta">
                          <span className="steer-pill">
                            ✨ Steered (k=
                            {m.trace?.variants?.length ?? config.k})
                          </span>
                          <span className="click-hint">Inspect Trace</span>
                        </div>
                      )}
                    </div>
                    {isAssistant && (
                      <div className="correction-row">
                        {flagged[i] ? (
                          <span className="correction-done" title="Stored as a negative training example">
                            ✗ marked wrong
                          </span>
                        ) : (
                          <button
                            className="correction-btn"
                            onClick={(e) => {
                              e.stopPropagation();
                              void markWrong(i);
                            }}
                            title="Flag this reply as wrong — stored as an asymmetric negative training example (corrections, not approval)"
                          >
                            ✗ wrong
                          </button>
                        )}
                      </div>
                    )}
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
                    <div className="thinking-text">
                      Steering candidate responses...
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="composer-area">
            {/* Collapsible Tuning Panel */}
            <div
              className={`config-drawer ${configExpanded ? "expanded" : ""}`}
            >
              <button
                className="drawer-toggle"
                onClick={() => setConfigExpanded(!configExpanded)}
              >
                <span className="toggle-icon">⚙️</span>
                <span className="toggle-label">Tuning Parameters</span>
                <span className="toggle-summary">
                  {!configExpanded &&
                    (() => {
                      const profileKey = getActiveProfileKey(config);
                      const profileLabel =
                        profileKey === "mediumLow"
                          ? "Medium Low"
                          : profileKey.charAt(0).toUpperCase() +
                            profileKey.slice(1);
                      return `Profile: ${profileLabel} | k=${config.k} | novelty=${config.novelty_weight.toFixed(2)} | temp=${config.temperature.toFixed(2)}`;
                    })()}
                </span>
                <span className="chevron">{configExpanded ? "▼" : "▲"}</span>
              </button>
              {configExpanded && (
                <Controls
                  config={config}
                  setConfig={handleConfigChange}
                  disabled={streaming}
                />
              )}
            </div>

            <div className="composer">
              <textarea
                value={input}
                placeholder={
                  streaming
                    ? "Waiting for response..."
                    : "Ask something creative..."
                }
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
          <TracePanel
            trace={displayedTrace}
            streaming={streaming}
            configUsed={displayedConfig}
            onApplyConfig={handleConfigChange}
          />
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
      return {
        ...t,
        frontier: ev.frontier,
        selected: ev.index,
        floorMet: ev.floor_met ?? true,
        chosenQuality: ev.chosen_quality ?? null,
      };
    case "response":
      return { ...t, done: true, synthesized: ev.synthesized ?? false };
    case "controller":
      return { ...t, controller: ev };
    case "synthesis":
      return {
        ...t,
        synthesisSources: ev.sources,
        synthesisCollapsed: ev.collapsed_to_modal ?? false,
      };
    case "grounding":
      return {
        ...t,
        grounding: {
          memory: ev.memory,
          tools: ev.tools,
          snippets: ev.snippets,
        },
      };
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
  const [activeTab, setActiveTab] = useState<
    "core" | "weights" | "advanced" | "budgets"
  >("core");

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
              {key === "mediumLow"
                ? "Medium Low"
                : key.charAt(0).toUpperCase() + key.slice(1)}
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

      {/* Tabs list */}
      <div className="control-tabs">
        <button
          type="button"
          className={`tab-btn ${activeTab === "core" ? "active" : ""}`}
          onClick={() => setActiveTab("core")}
        >
          Core Settings
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === "weights" ? "active" : ""}`}
          onClick={() => setActiveTab("weights")}
        >
          Weights
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === "advanced" ? "active" : ""}`}
          onClick={() => setActiveTab("advanced")}
        >
          Chains & Funnels
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === "budgets" ? "active" : ""}`}
          onClick={() => setActiveTab("budgets")}
        >
          Token Budgets
        </button>
      </div>

      <div className="controls-grid">
        {activeTab === "core" && (
          <>
            <div className="control-item">
              <div className="control-header">
                <label htmlFor="variants-k">Brainstorm Size (k)</label>
                <span className="control-val">{config.k}</span>
              </div>
              <input
                id="variants-k"
                type="range"
                min={3}
                max={15}
                step={1}
                value={config.k}
                onChange={setNum("k")}
                disabled={disabled}
              />
              <div className="control-desc">
                Number of unique candidates to generate.
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="temperature">Creativity (Temp)</label>
                <span className="control-val">
                  {config.temperature.toFixed(2)}
                </span>
              </div>
              <input
                id="temperature"
                type="range"
                min={0}
                max={2.0}
                step={0.05}
                value={config.temperature}
                onChange={setNum("temperature")}
                disabled={disabled}
              />
              <div className="control-desc">
                Randomness of alternative brainstorming.
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="quality-floor">Quality Floor</label>
                <span className="control-val">
                  {config.convergent_floor.toFixed(2)}
                </span>
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
              <div className="control-desc">
                Minimum rubric grade required to select.
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="seed">Seed</label>
                <span className="control-val">
                  {(config.seed ?? 0) === 0 ? "Random" : config.seed}
                </span>
              </div>
              <div className="seed-row">
                <input
                  id="seed"
                  type="number"
                  min={0}
                  step={1}
                  value={config.seed ?? 0}
                  onChange={setNum("seed")}
                  disabled={disabled}
                />
                <button
                  type="button"
                  className="seed-btn"
                  onClick={() =>
                    setConfig({
                      ...config,
                      seed: Math.floor(Math.random() * 1_000_000) + 1,
                    })
                  }
                  disabled={disabled}
                  title="Pick a random fixed seed"
                >
                  🎲
                </button>
                <button
                  type="button"
                  className="seed-btn"
                  onClick={() => setConfig({ ...config, seed: 0 })}
                  disabled={disabled}
                  title="0 = fully random"
                >
                  Clear
                </button>
              </div>
              <div className="control-desc">
                0 = random each run. Set a value to reproduce same generation.
              </div>
            </div>
          </>
        )}

        {activeTab === "weights" && (
          <>
            <div className="control-item">
              <div className="control-header">
                <label htmlFor="novelty-weight">Novelty Weight</label>
                <span className="control-val">
                  {config.novelty_weight.toFixed(2)}
                </span>
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
              <div className="control-desc">
                Preference for novelty vs. quality score.
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="coherence-weight">Coherence Weight</label>
                <span className="control-val">
                  {config.coherence_weight.toFixed(2)}
                </span>
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
              <div className="control-desc">
                Preference for stable idea attractor basin depth.
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="openness-weight">Openness Weight</label>
                <span className="control-val">
                  {config.openness_weight.toFixed(2)}
                </span>
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
              <div className="control-desc">
                Preference for counterfactual branching openness score.
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="originality-weight">Originality Weight</label>
                <span className="control-val">
                  {(config.originality_weight ?? 0).toFixed(2)}
                </span>
              </div>
              <input
                id="originality-weight"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={config.originality_weight ?? 0}
                onChange={setNum("originality_weight")}
                disabled={disabled}
              />
              <div className="control-desc">
                Penalize clichés and memorized jokes. Measures judge
                originality.
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="surprise-weight">Surprise Weight</label>
                <span className="control-val">
                  {(config.surprise_weight ?? 0).toFixed(2)}
                </span>
              </div>
              <input
                id="surprise-weight"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={config.surprise_weight ?? 0}
                onChange={setNum("surprise_weight")}
                disabled={disabled}
              />
              <div className="control-desc">
                Reward low token-confidence (recitation filter). Sequential
                generation.
              </div>
            </div>
          </>
        )}

        {activeTab === "advanced" && (
          <>
            <div className="control-item">
              <div className="control-header">
                <label htmlFor="breadth-k">
                  Breadth Candidates (breadth_k)
                </label>
                <span className="control-val">
                  {config.breadth_k === 0 ? "Off" : config.breadth_k}
                </span>
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
              <div className="control-desc">
                Generate this many total candidates (0 to disable).
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="prime-n">Funnel Primes (prime_n)</label>
                <span className="control-val">
                  {config.prime_n === 0 ? "Off" : config.prime_n}
                </span>
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
              <div className="control-desc">
                Keep this many diverse primes from candidate pool (0 to
                disable).
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="refine-passes">
                  Refine Passes (Chain Depth)
                </label>
                <span className="control-val">
                  {(config.refine_passes ?? 0) === 0
                    ? "Off"
                    : config.refine_passes}
                </span>
              </div>
              <input
                id="refine-passes"
                type="range"
                min={0}
                max={4}
                step={1}
                value={config.refine_passes ?? 0}
                onChange={setNum("refine_passes")}
                disabled={disabled}
              />
              <div className="control-desc">
                Critique → revise → re-score each prime. Collapsed iterations
                are rejected.
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="openness-branches">Openness Branches</label>
                <span className="control-val">
                  {config.openness_branches === 0
                    ? "Off"
                    : config.openness_branches}
                </span>
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
              <div className="control-desc">
                Number of continuation branches to probe (0 to disable).
              </div>
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
                  <span className="toggle-label-text">
                    Deepen Candidates (Branch)
                  </span>
                </label>

                <label className="toggle-container" htmlFor="synthesize-check">
                  <input
                    id="synthesize-check"
                    type="checkbox"
                    checked={config.synthesize}
                    onChange={setBool("synthesize")}
                    disabled={disabled}
                  />
                  <span className="toggle-label-text">
                    Synthesize Frontier (Merge)
                  </span>
                </label>

                <label className="toggle-container" htmlFor="trajectory-check">
                  <input
                    id="trajectory-check"
                    type="checkbox"
                    checked={config.trajectory ?? false}
                    onChange={setBool("trajectory")}
                    disabled={disabled}
                  />
                  <span className="toggle-label-text">
                    Cluster-Aware Trajectory
                  </span>
                </label>
              </div>
              <div className="control-desc" style={{ marginTop: "4px" }}>
                Trajectory waves cover new semantic regions (semantic entropy).
                Refine uses a guided critique chain.
              </div>
            </div>
          </>
        )}

        {activeTab === "budgets" && (
          <>
            <div className="control-item">
              <div className="control-header">
                <label htmlFor="response-tokens">Max Response Length</label>
                <span className="control-val">
                  {(config.response_tokens ?? 0) === 0
                    ? "Unlimited"
                    : config.response_tokens}
                </span>
              </div>
              <input
                id="response-tokens"
                type="range"
                min={0}
                max={16384}
                step={256}
                value={config.response_tokens ?? 0}
                onChange={setNum("response_tokens")}
                disabled={disabled}
              />
              <div className="control-desc">
                Token ceiling for the final answer. 0 = unlimited (the model's
                own EOS).
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="brainstorm-tokens">Brainstorm Budget</label>
                <span className="control-val">
                  {(config.brainstorm_tokens ?? 700) === 0
                    ? "Unlimited"
                    : (config.brainstorm_tokens ?? 700)}
                </span>
              </div>
              <input
                id="brainstorm-tokens"
                type="range"
                min={0}
                max={4000}
                step={100}
                value={config.brainstorm_tokens ?? 700}
                onChange={setNum("brainstorm_tokens")}
                disabled={disabled}
              />
              <div className="control-desc">
                Shared token budget for the numbered seed list (all candidates).
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="modal-tokens">Modal Baseline Budget</label>
                <span className="control-val">
                  {(config.modal_tokens ?? 256) === 0
                    ? "Unlimited"
                    : (config.modal_tokens ?? 256)}
                </span>
              </div>
              <input
                id="modal-tokens"
                type="range"
                min={0}
                max={2048}
                step={64}
                value={config.modal_tokens ?? 256}
                onChange={setNum("modal_tokens")}
                disabled={disabled}
              />
              <div className="control-desc">
                Token cap for greedy restatement (modal). Kept short by default.
              </div>
            </div>

            <div className="control-item">
              <div className="control-header">
                <label htmlFor="branch-tokens">
                  Branching Cap (branch_tokens)
                </label>
                <span className="control-val">
                  {(config.branch_tokens ?? 0) === 0
                    ? "Unlimited"
                    : config.branch_tokens}
                </span>
              </div>
              <input
                id="branch-tokens"
                type="range"
                min={0}
                max={4000}
                step={100}
                value={config.branch_tokens ?? 0}
                onChange={setNum("branch_tokens")}
                disabled={disabled}
              />
              <div className="control-desc">
                Token cap for each deepened "prime" candidate. 0 = unlimited.
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
