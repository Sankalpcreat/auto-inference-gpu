"""Turns `claude -p --output-format stream-json` into a readable live log (one line per step)."""
import json, sys, time
for line in sys.stdin:
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        continue
    t = time.strftime("%H:%M:%S")
    if ev.get("type") == "assistant":
        for c in ev["message"].get("content", []):
            if c.get("type") == "text" and c["text"].strip():
                print(f"{t}  💬 {c['text'].strip()[:400]}", flush=True)
            elif c.get("type") == "tool_use":
                i = c.get("input", {})
                arg = i.get("command") or i.get("file_path") or i.get("query") or i.get("url") or ""
                print(f"{t}  🔧 {c['name']}: {str(arg)[:200]}", flush=True)
    elif ev.get("type") == "result":
        print(f"{t}  ✅ DONE ({ev.get('duration_ms', 0) / 1000:.0f} s)\n{ev.get('result', '')}", flush=True)
