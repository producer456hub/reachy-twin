"""Standalone dual-brain voice loop (no web panel): mic -> Whisper -> brain -> Kokoro.

Thin CLI runner over the same RobotHub the panel uses -- one code path for capture,
wake-word routing, and speech. Run with the daemon already running:

    python -m twin.app
"""
import time

from twin.hub import RobotHub


def main():
    hub = RobotHub()
    hub.start()
    print(f"[ready] brains: {', '.join(hub.brains)} | active: {hub.active}")
    print(f"[mic] noise-gate = {hub._thresh:.4f}")
    hub.say("Hey, Maintop here. Say Claude to switch, or my name to come back. I'm listening.")
    hub.set_listening(True)
    try:
        while hub._listening:          # mic loop exits itself on the exit words
            time.sleep(0.2)
        print("[stopped] heard an exit word")
    except KeyboardInterrupt:
        print("\n[stopped]")
    finally:
        hub.shutdown()


if __name__ == "__main__":
    main()
