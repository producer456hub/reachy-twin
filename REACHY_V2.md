# Reachy Twin — v2.0 design

A feature-by-feature rebuild plan. Goal: keep everything that works, fix the
architecture that made it fragile, and make this **NVIDIA laptop** a first-class
host (local LLM brain + GPU STT) so the robot needs nothing on the network.

Written after a full read of `twin/` at commit `445bed3` (laptop clone).

---

## The one real problem

`twin/hub.py` is **1,274 lines** holding ~8 unrelated concerns. Everything else
in the package is 1–15 KB. The mess is concentrated there, and it has one root
cause:

**Too many writers to one piece of state — head/body orientation.**

These all command yaw/pitch, arbitrated only by timestamp gates and two locks:

| Writer | Method |
|---|---|
| turn-to-sound saccade | `_gaze_tick` |
| postural follow | `_follow_tick` |
| face tracking | `_face_track_tick` |
| idle breathing/recenter | `_idle_tick` |
| manual jog / look / center | `jog` / `look` / `center` → `_apply_pose` |
| emotion move while speaking | `_speak_now` |
| sleep / wake pose | `sleep` / `wake` |
| antenna flutter while thinking | `_antenna_flutter` |

They coordinate through `_move_until`, `_manual_until`, `_active_ts`,
`_follow_since`, `_pose`, `_lock`, plus re-reads of the servos because `_pose`
goes stale ("the axes the behavior layer moves behind `_pose`'s back"). That
juggling is the bug surface — every "untuned / flip the sign / fights the other
layer" note traces back to it.

**v2.0 fix:** one `MotionArbiter` that *owns* orientation and accepts prioritized
requests. Priority order: `manual > face > gaze-saccade > follow > idle`. Each
source just submits "I want to look here, at this priority"; the arbiter decides,
clamps to limits, and is the single caller of `goto_target` for orientation.
Delete `_move_until` / `_manual_until` / `_follow_since` / the servo re-reads.

---

## Proposed module layout

Split the god-object into owners, leave `hub.py` as a thin facade the panel/CLI
talk to:

```
twin/
  hub.py          # facade: wires the pieces, exposes chat()/say()/state()
  connection.py   # Supervisor: connect, auto-reconnect, daemon-kick, link liveness
  audio/
    pump.py       # single mic consumer + anti-echo ring buffer (the hard-won part)
    capture.py    # utterance segmentation (VAD)
    speech.py     # speech queue + worker + sentence chunking + TTS playback
  motion/
    arbiter.py    # NEW: single owner of head/body orientation (see above)
    gaze.py       # turn-to-sound (DoA) -> submits to arbiter
    face.py       # face tracking -> submits to arbiter
    idle.py       # aliveness -> submits to arbiter (lowest priority)
    expressions.py# emotions, dances, learned gestures (whole-body moves)
    poses.py      # sleep/wake, center, named rest poses
  brains/
    base.py       # _ChatBrain + history
    local.py      # NEW: Ollama (offline default)
    claude.py     # CLI (subscription) + API
    marcus.py     # vr-2 HTTP (optional)
    router.py     # wake-word routing, mood tags, action tags, prompt template
  nlu.py          # volume-command regexes (and any other text intents)
  server.py       # FastAPI panel (was panel.py)
  app.py          # standalone CLI loop
  config.py       # + host profiles (see below)
```

---

## Feature-by-feature decisions

Legend: **KEEP** = port as-is · **MOVE** = same logic, new home · **REDESIGN** =
change how it works · **NEW** · **CUT**

### Brains & conversation
| Feature | Decision | Notes |
|---|---|---|
| Dual-brain wake-word router | KEEP / MOVE → `brains/router.py` | sticky switch + per-brain voice is good |
| ClaudeCLIBrain (subscription, no key) | KEEP | session-resume design is right |
| ClaudeBrain (API) | KEEP | auto-picked only if key set |
| MarcusBrain (vr-2) | **DECISION** | tool-call queries still emit raw JSON; now that a **local brain** covers "offline," is Marcus worth keeping on the robot? Lean: make it opt-in, off by default |
| **Local Ollama brain** | **NEW** | offline default. Runs on *this* laptop's 4070. Becomes the always-available brain so the robot never needs the network |
| Mood tags `[happy]…` | KEEP / MOVE → router | verify a 7B follows the format as well as Claude; add a keyword fallback (already exists: `_emotion_for`) |
| Action tags `[dance]/[gesture]/[look]` | KEEP / MOVE → router | brittle on small models — test tag adherence; if the Ollama model supports tools, consider function-calling instead |
| Tool-call JSON guard | KEEP | only needed while Marcus stays |

### Audio
| Feature | Decision | Notes |
|---|---|---|
| Mic pump (single consumer + anti-echo) | KEEP / MOVE → `audio/pump.py` | the most valuable, most fragile code. Isolate + document the invariant. **Do not touch casually** |
| Energy-gate VAD + `calibrate_floor` | **REDESIGN** | hand-tuned RMS needed re-tuning per room (0.09 vs 0.0126). Replace with **silero-vad** or webrtcvad — robust across rooms, no magic numbers |
| STT faster-whisper | **REDESIGN** | currently `device="cpu", int8`. On this laptop move to **`device="cuda", float16`** and bump to `small.en`/`distil-small.en` — big latency win, GPU is sitting idle |
| TTS Kokoro | KEEP | solid, multi-voice |
| Piper fallback | **CUT** | dead weight now that Kokoro is primary |
| Speech queue + sentence chunking + motion overlap | KEEP / MOVE → `audio/speech.py` | "start talking after sentence 1" + overlap-with-emotion is good design |

### Motion
| Feature | Decision | Notes |
|---|---|---|
| **Motion arbiter** | **NEW** | the core fix — single owner of orientation, priority queue |
| Turn-to-sound (DoA gaze) | KEEP / REDESIGN → submits to arbiter | logic stays; stops fighting other layers by construction. Still needs live sign/deadzone tuning |
| Postural follow | KEEP / MOVE → arbiter | "body squares up under a held gaze" — nice, keep |
| Face-track (Haar) | KEEP, **platform-gated** | **dead on Windows** (camera IPC is Linux-only). Works on Mac/Linux host. Optionally upgrade Haar→mediapipe there |
| Idle aliveness | KEEP / MOVE → `motion/idle.py` | cheap, high "alive" payoff; lowest arbiter priority |
| Emotions + dances | KEEP / MOVE → `expressions.py` | |
| Learned gestures (hand-guided record/replay) | KEEP / MOVE → `expressions.py` | genuinely great, low cost |
| Antenna flutter while thinking | KEEP | small, good "I'm on it" cue |
| Sleep / wake | KEEP / MOVE → `motion/poses.py` | snapshot+restore is right |
| Manual jog / look / center | KEEP / MOVE → arbiter (highest priority) | replaces the `_manual_until` hack |

### Camera & vision
| Feature | Decision | Notes |
|---|---|---|
| Robot camera → JPEG | KEEP, platform-gated | **None on Windows** (GStreamer IPC is Linux-only) |
| iPad frame relay | KEEP | the Windows/headless workaround |
| Face detection | KEEP (Mac/Linux) | see face-track |

### Plumbing
| Feature | Decision | Notes |
|---|---|---|
| Supervisor / auto-reconnect | KEEP / MOVE → `connection.py` | daemon-kick is macOS-only (`launchctl`); add Windows/Linux equivalents or no-op cleanly |
| Volume-command NLU | KEEP / MOVE → `nlu.py` | the regex pile lives off the hub |
| FastAPI panel | KEEP / MOVE → `server.py` | already thin; maybe split route groups |
| Standalone CLI (`app.py`) | KEEP | thin runner over the hub |
| **Host profiles** | **NEW** in `config.py` | one place declaring per-host caps: `laptop` (Windows, GPU STT, **no camera**, local brain), `vr-2` (local brain, Marcus localhost), `mac` (camera + face-track + robot). Kills the scattered `sys.platform` checks |

---

## What "running on this laptop" actually means

This ASUS laptop = RTX 4070 **8GB** + Ryzen AI 9 HX 370 + 31 GB RAM, Windows.

- ✅ **Local brain** (Ollama, ~7B 4-bit) — fits 8 GB VRAM
- ✅ **GPU STT** (faster-whisper CUDA float16) — currently wasted on CPU
- ✅ TTS, audio loop, all conversation features
- ❌ **Camera + face-track** — Linux-only IPC; dead on this Windows host
- ➡️ Motion/gaze/face testing still needs the **Mac host + physical robot**

So v2.0 splits cleanly: **brain + audio + STT work** happens and is testable
here on the laptop; **motion/camera work** lands on the Mac host (which is also
~13 commits ahead — reconcile that first).

---

## Suggested build order

1. **Reconcile repos** — Mac is ahead of this laptop clone; pick the source of truth.
2. **Local brain** (`brains/local.py` + Ollama) — today's task; immediate payoff, testable on the laptop with no robot.
3. **GPU STT** — one-line-ish change, big latency win, testable on the laptop.
4. **Real VAD** — robustness, removes the per-room tuning.
5. **Module split** — mechanical, low-risk, do it incrementally.
6. **Motion arbiter** — the real redesign; do last, on the Mac host with the robot, because it needs live tuning.

Items 2–4 need no robot and run entirely on this machine.
</content>
</invoke>
