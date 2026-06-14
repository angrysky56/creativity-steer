import { useRef, useState } from "react";
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
  const abortRef = useRef<AbortController | null>(null);

  async function send() {
    const message = input.trim();
    if (!message || streaming) return;
    const history = messages;
    setMessages([...history, { role: "user", content: message }]);
    setInput("");
    setTrace(emptyTrace());
    setStreaming(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      for await (const ev of streamChat(message, history, config, ctrl.signal)) {
        setTrace((t) => reduce(t, ev));
        if (ev.type === "response") {
          setMessages((m) => [...m, { role: "assistant", content: ev.text }]);
        } else if (ev.type === "error") {
          setMessages((m) => [
            ...m,
            { role: "assistant", content: `⚠️ ${ev.message}` },
          ]);
        }
      }
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `⚠️ error: ${String(err)}` },
      ]);
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>creativity-steer</h1>
        <span className="subtitle">brainstorm → score novelty + quality → select</span>
      </header>
      <div className="layout">
        <section className="chat-col">
          <div className="messages">
            {messages.length === 0 && (
              <div className="hint">Ask anything. The right panel shows how the
              reply was chosen.</div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <div className="bubble">{m.content}</div>
              </div>
            ))}
            {streaming && <div className="msg assistant"><div className="bubble thinking">thinking…</div></div>}
          </div>
          <Controls config={config} setConfig={setConfig} disabled={streaming} />
          <div className="composer">
            <textarea
              value={input}
              placeholder="Type a message…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
            />
            <button onClick={() => void send()} disabled={streaming || !input.trim()}>
              Send
            </button>
          </div>
        </section>
        <section className="trace-col">
          <TracePanel trace={trace} streaming={streaming} />
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
    <div className="controls">
      <label>variants k <input type="range" min={3} max={8} step={1} value={config.k} onChange={set("k")} disabled={disabled} /><b>{config.k}</b></label>
      <label>novelty wt <input type="range" min={0} max={1} step={0.05} value={config.novelty_weight} onChange={set("novelty_weight")} disabled={disabled} /><b>{config.novelty_weight.toFixed(2)}</b></label>
      <label>quality floor <input type="range" min={0} max={1} step={0.05} value={config.convergent_floor} onChange={set("convergent_floor")} disabled={disabled} /><b>{config.convergent_floor.toFixed(2)}</b></label>
      <label>temperature <input type="range" min={0} max={1.5} step={0.05} value={config.temperature} onChange={set("temperature")} disabled={disabled} /><b>{config.temperature.toFixed(2)}</b></label>
    </div>
  );
}
