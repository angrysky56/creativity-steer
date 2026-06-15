import type { ChatMessage, Config, TraceEvent } from "./types";

// Stream the chat trace from the FastAPI SSE endpoint. Native EventSource is
// GET-only, so we POST and parse the SSE frames off the fetch body stream.
export async function* streamChat(
  message: string,
  history: ChatMessage[],
  config: Config,
  signal: AbortSignal,
): AsyncGenerator<TraceEvent> {
  const sanitizedHistory = history.map(({ role, content }) => ({ role, content }));
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history: sanitizedHistory, ...config }),
    signal,
  });
  if (!res.body) throw new Error("no response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      const payload = line.slice(6).trim();
      if (!payload || payload === "{}") continue;
      try {
        yield JSON.parse(payload) as TraceEvent;
      } catch {
        // ignore malformed frame
      }
    }
  }
}
