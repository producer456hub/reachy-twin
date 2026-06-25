"""RobotHub: one ReachyMini connection shared by the voice loop and the web panel.

Thread-safety: motion/camera and audio each have their own lock. The mic has exactly
ONE consumer: the always-on pump thread (_mic_pump), which drains the daemon's recording
pipeline into a small ring buffer. While he speaks (_speaking set) the pump DISCARDS
samples, so he can't hear himself; _capture() reads only from the ring buffer. The pump
must never stop while connected -- an unconsumed pipeline backs up and drops samples
("Can't record audio fast enough"). Brain calls (Claude CLI / Marcus HTTP) are not locked.

Speech is queued: chat()/say() return as soon as the reply text is known; a single
worker thread synthesizes + plays in order, so HTTP callers never wait out the audio.

The hub may start with no robot attached (panel retries in the background); every
method that touches self.mini degrades gracefully until the daemon connects.
"""
import glob
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque

import numpy as np
from reachy_mini import ReachyMini

from twin.stt import STT
from twin.tts import KokoroTTS
from twin.voice import (
    build_brains, detect_switch, strip_wake, calibrate_floor,
    _rms, _mono, SR, EXIT_WORDS, SPEECH_START, SILENCE_HANG, MIN_CHUNKS,
)
from twin.brains import strip_mood, MOOD_TO_EMOTION

DAEMON = "http://localhost:8000"
BODY_YAW_MAX = 2.7   # rad, mechanical-ish limit used everywhere we command the body
GESTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gestures")

# inline action tags the LLM brain may emit ([dance], [gesture:hi], [look:left], ...)
ACTION_RX = re.compile(r"\[(dance|gesture|look)(?::([a-zA-Z0-9_ \-]+))?\]", re.I)
LOOK_PRESETS = {"left": (28, 0), "right": (-28, 0), "up": (0, -15), "down": (0, 18), "center": (0, 0)}


class RobotHub:
    def __init__(self):
        self.stt = STT()
        self.tts = KokoroTTS()
        self.brains, self.voices = build_brains()
        self.active = "maintop"          # this host's own local brain is the default
        self.mini = None
        self.last_error = None     # last robot-connect failure (shown in /api/state)
        self._lock = threading.Lock()          # serializes SERVO/motion + camera access
        self._audio_lock = threading.Lock()    # serializes the AUDIO pipeline (speaker/mic), separate from motion
        self._speaking = threading.Event()      # set while a reply is playing out loud
        self._connect_lock = threading.Lock()  # supervisor + /api/robot/reconnect must not double-connect
        # mic ring buffer: the pump is the ONLY reader of the SDK mic; _capture reads here
        self._mic_buf = deque(maxlen=512)      # ~10ms chunks -> a few seconds of backlog max
        self._pump_stop = None                 # per-connection stop event for the mic pump
        self._listening = False
        self._stop = threading.Event()
        self._thread = None
        self._thresh = 0.02
        self.log = deque(maxlen=100)
        self.emotions = None       # RecordedMoves (lazy)
        self._dances = None        # list[str]
        # autonomous behaviors
        self.behaviors = {"turn_to_sound": False, "face_track": False,
                          "emotions_on_cue": False, "idle_motion": False}
        self._behavior_stop = threading.Event()
        self._behavior_thread = None
        self.doa_sign = -1.0       # flip if he turns the wrong way (tuned live)
        # gaze controller (single owner of neck/body orientation):
        # turn-to-sound saccades the head fast; the follow layer then walks the body
        # under the gaze in ONE smooth move with the head counter-rotating.
        self._doa_hist = deque(maxlen=3)   # recent (t, absolute target yaw) candidates
        self._move_until = 0.0             # wall-clock end of in-flight goto; don't interrupt
        self._manual_until = 0.0           # behaviors stand down until here after a manual jog/look/center
        self._follow_since = None          # dwell timer before the body commits
        # idle "aliveness" layer (gentle breathing + slow recenter when nothing else is happening)
        self._active_ts = 0.0              # last real activity (gaze/face/follow/manual/speech)
        self._last_idle = 0.0              # throttle for the idle tick
        self._asleep = False               # sleep/wake: behaviors+pose parked, restored on wake
        self._sleep_state = None           # snapshot taken at sleep() to restore at wake()
        self._vol_cache = None     # cache slow /api/volume/current
        self._vol_ts = 0.0
        self._vol_refreshing = False
        self._last_face = 0.0
        self._face_seen_ts = 0.0   # last time a face was actually detected
        self._cached_mode = "?"    # /api/motors/status is slow (~2s) -> cache it
        self._mode_ts = 0.0
        self._servo_cache = {"body_yaw": None, "head_yaw": None, "head_pitch": None,
                             "antenna_left": None, "antenna_right": None}
        self._cascade = None       # opencv face detector (lazy)
        self._pose = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "body": 0.0, "ant": 0.0}  # degrees
        # iPad-relayed camera frame (used when robot camera is unavailable, e.g. Windows host)
        self._ipad_jpeg = None
        self._ipad_jpeg_ts = 0.0
        # ordered speech queue -> single worker, so replies don't block HTTP responses
        self._speech_q = queue.Queue()
        self._speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._speech_thread.start()
        # supervisor (auto-reconnect across robot replugs / daemon restarts)
        self._supervisor_thread = None
        self._supervisor_stop = threading.Event()
        self._last_kick = 0.0
        self._thinking = threading.Event()   # set while a brain works; drives antenna flutter
        self._recording_gesture = False      # behaviors pause while limp for hand-guiding
        os.makedirs(GESTURE_DIR, exist_ok=True)

    def push_ipad_frame(self, jpeg_bytes):
        if jpeg_bytes:
            self._ipad_jpeg = jpeg_bytes
            self._ipad_jpeg_ts = time.time()

    def get_ipad_jpeg(self, max_age_sec=10.0):
        if not self._ipad_jpeg:
            return None
        if time.time() - self._ipad_jpeg_ts > max_age_sec:
            return None
        return self._ipad_jpeg

    # ---------- lifecycle ----------
    def start(self):
        with self._connect_lock:
            self._start_locked()

    def _start_locked(self):
        if self.mini is not None:  # supervisor + reconnect raced; the link is already up
            return
        # On Windows the LOCAL/GStreamer-IPC camera path is broken, so let the env
        # force webrtc (cross-platform). Default behavior unchanged for Linux/macOS.
        backend = os.environ.get("REACHY_MEDIA_BACKEND", "default")
        mini = ReachyMini(media_backend=backend)
        mini.__enter__()
        try:
            mini.media.start_recording()
            mini.media.start_playing()
            self._thresh = calibrate_floor(mini)
        except Exception:
            mini.__exit__(None, None, None)
            raise
        self.mini = mini           # publish only once fully usable
        self.last_error = None
        self._start_mic_pump(mini)
        try:
            self._load_moves()
        except Exception as e:
            self._log("system", f"moves library unavailable: {e}")
        self._log("system", f"online - brains: {', '.join(self.brains)} | gate {self._thresh:.4f}")
        # tell the Claude brain what physical actions it can call
        from twin import brains as _brains
        _brains.set_actions_hint(self._dances or [], self.list_gestures())
        # always-on by default: ears find him a face, eyes hold it, moods get acted out
        self.set_behavior("face_track", True)
        self.set_behavior("turn_to_sound", True)
        self.set_behavior("emotions_on_cue", True)
        self.set_behavior("idle_motion", True)

    def shutdown(self):
        self._supervisor_stop.set()
        self._behavior_stop.set()
        self.set_listening(False)
        self._teardown_mini()

    @property
    def robot_connected(self):
        return self.mini is not None

    def _teardown_mini(self):
        mini, self.mini = self.mini, None
        if self._pump_stop is not None:
            self._pump_stop.set()       # the pump dies with its connection
        if mini is None:
            return
        try:
            mini.media.stop_recording()
            mini.media.stop_playing()
            mini.__exit__(None, None, None)
        except Exception:
            pass

    # ---------- supervision (auto-reconnect) ----------
    # Owns the robot link end to end: initial connect, reconnect after the robot
    # is replugged, and bouncing a "robot-less" daemon (one that started while
    # the robot was unplugged -- it serves HTTP but has no motors, and SDK
    # clients can't attach to it).
    SUPERVISE_PERIOD = 5.0
    KICK_COOLDOWN = 60.0

    def start_supervisor(self):
        if self._supervisor_thread is None or not self._supervisor_thread.is_alive():
            self._supervisor_stop.clear()
            self._supervisor_thread = threading.Thread(target=self._supervise, daemon=True)
            self._supervisor_thread.start()

    def stop_supervisor(self):
        self._supervisor_stop.set()

    def _daemon_alive(self):
        try:
            urllib.request.urlopen(DAEMON + "/api/daemon/status", timeout=2)
            return True
        except Exception:
            return False

    def _serial_present(self):
        if sys.platform == "darwin":
            return bool(glob.glob("/dev/cu.usbmodem*"))
        return True   # elsewhere, don't gate on it

    def _kick_daemon(self, force=False):
        """Restart the daemon's launchd service (macOS host only)."""
        now = time.time()
        if not force and now - self._last_kick < self.KICK_COOLDOWN:
            return False
        if sys.platform != "darwin":
            return False
        self._last_kick = now
        self._log("system", "daemon has no robot -- restarting it")
        try:
            subprocess.run(["launchctl", "kickstart", "-k",
                            f"gui/{os.getuid()}/com.legionstage.reachy-daemon"],
                           timeout=10, capture_output=True)
            return True
        except Exception as e:
            self._log("system", f"daemon restart failed: {e}")
            return False

    def _supervise(self):
        connect_fails = 0
        link_fails = 0
        cam_fails = 0
        cam_ok_once = False   # only treat a dead camera as fatal if it worked this session
        while not self._supervisor_stop.is_set():
            self._supervisor_stop.wait(self.SUPERVISE_PERIOD)
            if self._supervisor_stop.is_set():
                break
            if self.mini is None:
                try:
                    self.start()
                    connect_fails = 0
                except Exception as e:
                    self.last_error = str(e)
                    connect_fails += 1
                    # daemon serving + robot USB present, yet clients can't attach
                    # -> the daemon started while the robot was away; bounce it
                    if connect_fails >= 2 and self._serial_present() and self._daemon_alive():
                        self._kick_daemon()
                continue
            # link liveness: cheap joint read; skip the check while he's speaking
            if not self._lock.acquire(timeout=1.5):
                continue
            try:
                self.mini.get_current_joint_positions()
                link_fails = 0
            except Exception:
                link_fails += 1
            finally:
                self._lock.release()
            if link_fails >= 2:                 # two strikes -> rebuild the link
                self._log("system", "robot link stale -- reconnecting")
                link_fails = 0
                self._teardown_mini()
                continue
            # camera liveness: motors can survive a replug while the daemon's
            # video pipeline silently dies bound to the old USB device
            jpg = self.get_jpeg(quality=40)
            if jpg:
                cam_ok_once = True
                cam_fails = 0
            elif cam_ok_once:
                cam_fails += 1
                if cam_fails >= 4:              # ~20s of no frames after having worked
                    cam_fails = 0
                    self._log("system", "camera stream died -- restarting daemon")
                    if self._kick_daemon():
                        self._teardown_mini()   # rebuild our client against the new daemon

    def reconnect(self):
        """Client-requested: force-reestablish the robot link right now."""
        self._teardown_mini()
        try:
            self.start()
            return {"ok": True, "robot": "connected"}
        except Exception as e:
            self.last_error = str(e)
            if self._serial_present() and self._daemon_alive() and self._kick_daemon(force=True):
                note = "daemon restarting -- auto-reconnect will finish in ~30s"
            else:
                note = str(e)
            return {"ok": False, "robot": "disconnected", "note": note}

    # ---------- logging ----------
    def _log(self, who, text):
        self.log.append({"who": who, "text": text, "t": time.time()})

    # ---------- speech (queued) ----------
    def say(self, text, voice=None):
        self._enqueue_say(text, None, voice)

    def _enqueue_say(self, text, move, voice=None):
        if not text:
            return
        if self.mini is None:
            self._log("system", "(robot offline -- reply not spoken)")
            return
        self._speech_q.put((text, move, voice or self.voices.get(self.active, "af_heart")))

    def _enqueue_move(self, move):
        """Queue a motion-only item (plays after any queued speech finishes)."""
        if move is not None and self.mini is not None:
            self._speech_q.put((None, move, None))

    def _speech_worker(self):
        while True:
            text, move, voice = self._speech_q.get()
            try:
                if text:
                    self._speak_now(text, move, voice)
                elif move is not None:
                    with self._lock:
                        self.mini.play_move(move)
            except Exception as e:
                self._log("system", f"speech error: {e}")

    @staticmethod
    def _sentences(text, min_len=24):
        """Split a reply into speakable chunks; merge fragments so chunks stay natural."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        out = []
        for p in parts:
            if not p:
                continue
            if out and (len(out[-1]) < min_len or len(p) < 12):
                out[-1] += " " + p
            else:
                out.append(p)
        return out or [text]

    def _speak_now(self, text, move, voice):
        """Speak while an (optional) emotion move plays concurrently -- like real body language.

        Synthesis is sentence-chunked: he starts talking after the FIRST sentence is
        ready instead of waiting for the whole reply, and later sentences synthesize
        while earlier ones play. push_audio_sample is non-blocking and the daemon
        composes its audio-reactive wobble on top of the move's pose, so motion +
        speech layer instead of fighting.
        """
        chunks = self._sentences(text)
        # Speaking uses the AUDIO lock, not the motion lock, so a manual jog (and
        # the emotion move below) can run while he talks instead of freezing for
        # the whole reply. Synthesis is pure CPU -> never under any lock.
        self._speaking.set()
        self._active_ts = time.time()
        try:
            mt = None
            if move is not None:
                def _run():
                    try:
                        with self._lock:                         # motion lock; serializes with jogs
                            self.mini.play_move(move, sound=False)  # motion only; speech is the audio
                    except Exception:
                        pass
                mt = threading.Thread(target=_run, daemon=True)
                mt.start()
            # Each chunk starts playing at max(previous chunk's end, its push time) --
            # if synthesis runs slower than realtime, playback has gaps and ends later
            # than sum-of-durations. Track the real end so _speaking covers ALL of it
            # (the mic pump discards while _speaking; clearing early = echo).
            play_end = None
            for ch in chunks:
                samples = self.tts.synth(ch, voice=voice)        # CPU, outside the lock
                with self._audio_lock:
                    self.mini.media.push_audio_sample(samples)   # queues; playback is continuous
                pushed = time.time()
                base = pushed if (play_end is None or play_end < pushed) else play_end
                play_end = base + len(samples) / SR
            while play_end is not None and time.time() < play_end + 0.3:
                time.sleep(0.02)                 # mic discard is the pump's job now
            if mt is not None:
                mt.join(timeout=4)
        finally:
            self._speaking.clear()
            self._active_ts = time.time()

    # ---------- chat ----------
    THINK_FILLERS = ["Hmm.", "Hm, let me think.", "Mmm.", "Oh!", "Good question."]

    def _think_cue(self, spoken):
        """Instant acknowledgment while the brain works: antennas perk up and
        FLUTTER until the reply is ready (visible 'I'm on it'), plus a short
        verbal filler when the input came in by voice."""
        if self.mini is None:
            return
        if spoken:
            self._enqueue_say(random.choice(self.THINK_FILLERS), None)
        if not self._thinking.is_set():
            self._thinking.set()
            threading.Thread(target=self._antenna_flutter, daemon=True).start()

    def _antenna_flutter(self):
        """Oscillate the antennas while `_thinking` is set, then settle back."""
        try:
            base = self._pose["ant"] + self.ANT_REST_DEG
            t0 = time.time()
            up = True
            while self._thinking.is_set() and time.time() - t0 < 30:   # safety cap
                a = base + (22 if up else 12)
                up = not up
                if self._lock.acquire(timeout=0.5):     # skip beats while he speaks
                    try:
                        self.mini.goto_target(antennas=np.deg2rad([a, a]), duration=0.15)
                    finally:
                        self._lock.release()
                time.sleep(0.18)
            for _ in range(10):                          # settle back -- retry through lock contention
                if self._lock.acquire(timeout=1.0):
                    try:
                        self.mini.goto_target(antennas=np.deg2rad([base, base]), duration=0.4)
                        break
                    finally:
                        self._lock.release()
                time.sleep(0.3)
        except Exception:
            pass

    def chat(self, text, brain=None, spoken=False):
        text = (text or "").strip()
        if not text:
            return {"brain": self.active, "reply": ""}
        if self._maybe_volume_command(text):
            return {"brain": self.active, "reply": "[volume adjusted]", "command": "volume"}
        switched = detect_switch(text, self.active)
        if switched and switched in self.brains:
            self.active = switched
        elif brain and brain in self.brains:
            self.active = brain
        msg = strip_wake(text) if switched else text
        self._log("you", text)
        self._think_cue(spoken)                  # he reacts NOW; the answer follows
        try:
            if not msg.strip():
                reply = f"{self.active.capitalize()} here. What's up?"
            else:
                reply = self.brains[self.active].reply(msg)
        except Exception as e:                   # an API/network error must not 500 the route
            self._log("system", f"brain '{self.active}' error: {e}")
            reply = "Sorry, my brain hiccuped for a second there -- say that again?"
        finally:
            self._thinking.clear()               # stop the flutter; time to talk
        mood, reply = strip_mood(reply)          # pull the brain's [mood] tag off the speech
        actions, reply = self._extract_actions(reply)
        self._log(self.active, reply)
        move = None
        if self.behaviors.get("emotions_on_cue"):
            emo = MOOD_TO_EMOTION.get(mood) if mood else self._emotion_for(reply)
            if emo:
                move = self._get_emotion_move(emo)
        self._enqueue_say(reply, move)           # speech is async; reply returns immediately
        self._run_actions(actions)               # looks happen now; dances/gestures queue after speech
        return {"brain": self.active, "reply": reply, "mood": mood,
                "actions": [f"{k}:{v}" if v else k for k, v in actions] or None}

    def set_brain(self, brain):
        if brain in self.brains:
            self.active = brain
        return self.active

    # ---------- Maintop model switching (swap which local model he thinks with) ----------
    def list_brain_models(self):
        b = self.brains.get("maintop")
        models = b.list_models() if hasattr(b, "list_models") else []
        return {"models": models, "active": getattr(b, "model", None)}

    def set_brain_model(self, model):
        b = self.brains.get("maintop")
        if not hasattr(b, "set_model"):
            return {"error": "local brain unavailable"}
        active = b.set_model(model)
        self._log("system", f"Maintop model -> {active}")
        return {"active": active}

    # ---------- listening ----------
    def set_listening(self, on):
        on = bool(on)
        if on and not self._listening:
            self._stop.set()                     # stop + join any straggler loop first
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=2)
            if self.mini is None:
                self._log("system", "(robot offline -- can't listen)")
                return False
            self._listening = True
            self._stop.clear()
            self._thread = threading.Thread(target=self._mic_loop, daemon=True)
            self._thread.start()
        elif not on and self._listening:
            self._listening = False
            self._stop.set()
        return self._listening

    def _mic_loop(self):
        while not self._stop.is_set():
            try:
                audio = self._capture()
                if audio is None:
                    continue
                text = self.stt.transcribe(audio)
                if not text or len(text) < 2:
                    continue
                if any(w in text.lower() for w in EXIT_WORDS):
                    self.say("Okay, going quiet. Bye.")
                    self._listening = False
                    self._stop.set()
                    break
                self.chat(text, spoken=True)
            except Exception as e:
                # one STT/capture hiccup must not silently end listening
                now = time.time()
                if now - getattr(self, "_last_mic_err", 0) > 30:
                    self._last_mic_err = now
                    self._log("system", f"mic loop error (continuing): {e}")
                time.sleep(0.3)

    # ---------- mic pump (single consumer of the robot mic) ----------
    # The daemon's recording pipeline NEEDS a constant consumer or it backs up and
    # drops samples ("Can't record audio fast enough" spam). This thread is that
    # consumer for the life of one connection. While he speaks, samples are
    # DISCARDED (anti-echo) -- the old design had _capture and the speech drain
    # racing for the same samples, so he could hear and transcribe himself.
    def _start_mic_pump(self, mini):
        self._mic_buf.clear()
        self._pump_stop = threading.Event()
        threading.Thread(target=self._mic_pump, args=(mini, self._pump_stop),
                         daemon=True).start()

    def _mic_pump(self, mini, stop):
        while not stop.is_set():
            try:
                with self._audio_lock:
                    s = mini.media.get_audio_sample()
            except Exception:
                time.sleep(0.1)                  # link hiccup; supervisor handles real death
                continue
            if s is None or len(s) == 0:
                time.sleep(0.005)                # don't spin a core on an idle pipeline
                continue
            if self._speaking.is_set() or not self._listening:
                self._mic_buf.clear()            # discard: his own voice / nobody listening
                continue
            self._mic_buf.append(_mono(s))

    def _capture(self, max_seconds=15.0):
        buf, in_speech, run, silence = [], False, 0, 0
        start = time.time()
        while not self._stop.is_set() and time.time() - start < max_seconds:
            if self._speaking.is_set():          # he's talking: drop any half-built
                buf, in_speech, run, silence = [], False, 0, 0   # utterance, start fresh after
                time.sleep(0.05)
                continue
            try:
                m = self._mic_buf.popleft()      # fed by the mic pump
            except IndexError:
                time.sleep(0.005)
                continue
            loud = _rms(m) > self._thresh
            if not in_speech:
                if loud:
                    run += 1
                    buf.append(m)
                    if run >= SPEECH_START:
                        in_speech = True
                else:
                    run, buf = 0, []
            else:
                buf.append(m)
                silence = silence + 1 if not loud else 0
                if silence >= SILENCE_HANG:
                    break
        if not in_speech or len(buf) < MIN_CHUNKS:
            return None
        return np.concatenate(buf).astype(np.float32)

    # ---------- expressive moves (emotions + dances) ----------
    def _load_moves(self):
        if self.emotions is None:
            from reachy_mini.motion.recorded_move import RecordedMoves
            self.emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        if self._dances is None:
            from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
            self._dances = list(AVAILABLE_MOVES)

    def list_moves(self):
        self._load_moves()
        return {"emotions": self.emotions.list_moves(), "dances": self._dances}

    def play(self, kind, name):
        if self.mini is None:
            return {"error": "robot not connected"}
        self._load_moves()
        try:
            if kind == "emotion":
                move = self.emotions.get(name)
            elif kind == "dance":
                from reachy_mini_dances_library import DanceMove
                move = DanceMove(name)
            else:
                return {"error": f"unknown kind {kind!r}"}
        except Exception as e:
            return {"error": f"{name}: {e}"}
        try:
            with self._lock:
                self.mini.play_move(move)
        except Exception as e:
            return {"error": f"{name}: {e}"}
        self._log("system", f"played {kind}: {name}")
        return {"ok": True, "kind": kind, "name": name}

    # ---------- autonomous behaviors ----------
    def set_behavior(self, name, on):
        if name in self.behaviors:
            self.behaviors[name] = bool(on)
        need_thread = (self.behaviors["turn_to_sound"] or self.behaviors["face_track"]
                       or self.behaviors["idle_motion"])
        if need_thread and (self._behavior_thread is None or not self._behavior_thread.is_alive()):
            self._behavior_stop.clear()
            self._behavior_thread = threading.Thread(target=self._behavior_loop, daemon=True)
            self._behavior_thread.start()
        elif not need_thread:
            self._behavior_stop.set()
        return self.behaviors

    FACE_HOLDS_GAZE_S = 2.5   # eyes own the head this long after seeing a face

    def _behavior_loop(self):
        while not self._behavior_stop.is_set() and (
                self.behaviors["turn_to_sound"] or self.behaviors["face_track"]
                or self.behaviors["idle_motion"]):
            if self._recording_gesture:          # hands-off while being hand-guided
                time.sleep(0.2)
                continue
            if self._speaking.is_set():           # let the emotion-move own the body while talking
                time.sleep(0.1)
                continue
            if time.time() < self._manual_until:  # a human just jogged/looked -- don't fight it
                time.sleep(0.06)
                continue
            try:
                if self.behaviors["face_track"]:
                    self._face_track_tick()
                # ears find, eyes hold: sound only steers when no face is in view,
                # so a noise can't yank his gaze off the person he's looking at.
                if self.behaviors["turn_to_sound"] and (
                        not self.behaviors["face_track"]
                        or time.time() - self._face_seen_ts > self.FACE_HOLDS_GAZE_S):
                    self._gaze_tick()
                self._follow_tick()     # always-on postural layer: body follows a held head turn
                if self.behaviors["idle_motion"]:
                    self._idle_tick()   # gentle breathing + recenter when nothing else is happening
            except Exception as e:
                # an SDK hiccup (e.g. goto_target TimeoutError) must never kill
                # this thread -- that silently disables every behavior until restart
                now = time.time()
                if now - getattr(self, "_last_behavior_err", 0) > 30:
                    self._last_behavior_err = now
                    self._log("system", f"behavior tick error (continuing): {e}")
                time.sleep(0.5)
            time.sleep(0.06)

    # ---------- gaze controller ----------
    # Head saccades fast toward a sound (like a startle/glance); the body then squares
    # up underneath in one slow move while the head counter-rotates to hold the gaze,
    # settling with a small head lead so he never looks bolt-straight (reads as alive).
    HEAD_LOOK_MAX = 0.6     # rad (~34 deg) the head will glance on its own
    SACCADE_S = 0.25        # fast head move
    DOA_STABLE_RAD = 0.17   # commit only when recent DoA estimates agree within ~10 deg
    FOLLOW_TRIGGER_DEG = 14 # head held past this -> body starts to follow
    LEAD_DEG = 5            # residual head lead left after settling
    FOLLOW_DWELL = 1.0      # seconds the head must stay off-center before the body commits
    FOLLOW_S = 1.2          # slow body move

    def _read_yaws(self):
        """Measured (body_yaw, head_yaw) in rad from the servos -- no dead reckoning."""
        b, y, _ = self._read_pose()
        return b, y

    def _read_pose(self):
        """Measured (body_yaw, head_yaw, head_pitch) in rad from the servos."""
        try:
            with self._lock:
                head, _ = self.mini.get_current_joint_positions()
                m = np.asarray(self.mini.get_current_head_pose(), dtype=float)
            yaw = float(np.arctan2(m[1, 0], m[0, 0]))
            pitch = float(-np.arcsin(max(-1.0, min(1.0, m[2, 0]))))
            return float(head[0]), yaw, pitch
        except Exception:
            return None, None, None

    def _gaze_tick(self):
        now = time.time()
        if now < self._move_until:           # let the in-flight move finish
            return
        try:
            with self._audio_lock:
                doa = self.mini.media.get_DoA()
        except Exception:
            return
        if not doa:
            return
        angle, speech = doa
        if not speech:
            return
        delta = self.doa_sign * (angle - (np.pi / 2))   # offset from current facing (rad)
        if abs(delta) < 0.2:                 # near-front jitter -> already looking there
            self._doa_hist.clear()
            return
        body, head = self._read_yaws()
        if body is None:
            return
        # DoA is relative to where he's facing NOW -> make it an absolute target,
        # so consecutive estimates can be compared/filtered instead of re-chased.
        self._doa_hist.append((now, body + head + delta))
        fresh = [t for ts, t in self._doa_hist if now - ts < 2.0]
        if len(fresh) < 2 or max(fresh) - min(fresh) > self.DOA_STABLE_RAD:
            return                           # wait for two agreeing estimates (kills jitter)
        target = float(np.mean(fresh))
        self._doa_hist.clear()
        from reachy_mini.utils import create_head_pose
        head_target = max(-self.HEAD_LOOK_MAX, min(self.HEAD_LOOK_MAX, target - body))
        pose = create_head_pose(yaw=float(np.rad2deg(head_target)), degrees=True)
        with self._lock:
            self.mini.goto_target(head=pose, duration=self.SACCADE_S)
        self._move_until = now + self.SACCADE_S + 0.05
        self._follow_since = now             # follow dwell starts at the glance
        self._active_ts = now                # real activity -> hold off idle motion

    def _follow_tick(self):
        now = time.time()
        if now < self._move_until:
            return
        body, head = self._read_yaws()
        if body is None:
            return
        head_deg = float(np.rad2deg(head))
        if abs(head_deg) <= self.FOLLOW_TRIGGER_DEG:     # comfortable -> nothing to do
            self._follow_since = None
            return
        if self._follow_since is None:                   # start the dwell timer
            self._follow_since = now
            return
        if now - self._follow_since < self.FOLLOW_DWELL:
            return
        self._follow_since = None
        # one smooth move: body squares up under the gaze while the head counter-rotates
        # in the SAME goto, so the gaze direction holds throughout.
        sign = 1.0 if head_deg > 0 else -1.0
        target_total = body + head
        new_head = sign * float(np.deg2rad(self.LEAD_DEG))
        new_body = max(-BODY_YAW_MAX, min(BODY_YAW_MAX, target_total - new_head))
        if abs(new_body - body) < np.deg2rad(2):         # body already there (or at limit)
            return
        new_head = target_total - new_body               # exact gaze hold after body clamp
        from reachy_mini.utils import create_head_pose
        with self._lock:
            if self.behaviors.get("face_track"):
                # the camera re-centers the head on its own; just bring the body around
                self.mini.goto_target(body_yaw=new_body, duration=self.FOLLOW_S)
            else:
                pose = create_head_pose(yaw=float(np.rad2deg(new_head)), degrees=True)
                self.mini.goto_target(head=pose, body_yaw=new_body, duration=self.FOLLOW_S)
        self._move_until = now + self.FOLLOW_S + 0.1
        self._active_ts = now

    # ---------- idle "aliveness" ----------
    # When nothing else is going on, a calm breathing-like sway keeps him from
    # looking switched-off, and he eases back toward neutral so he doesn't sit
    # frozen staring off-axis. Everything here is small, slow, and strictly gated
    # so it never fights gaze / face-track / follow / a manual jog / speech.
    IDLE_AFTER = 6.0           # seconds of calm before idle motion starts
    IDLE_SWAY_DEG = 1.6        # whole-body sway amplitude (kept tiny)
    IDLE_PITCH_DEG = 1.4       # head breathing-nod amplitude
    IDLE_YAW_DEG = 2.5         # slow lazy head-yaw wander
    IDLE_RECENTER = 0.06       # fraction of any off-center baseline shed per tick
    IDLE_TICK_S = 0.5          # ~2 Hz: gentle on the servos

    def _idle_tick(self):
        now = time.time()
        # Only when truly idle: not talking/thinking/being hand-guided, no move in
        # flight, past the manual-hold, and nothing has steered him recently.
        if (self._speaking.is_set() or self._thinking.is_set() or self._recording_gesture
                or now < self._move_until or now < self._manual_until
                or now - self._active_ts < self.IDLE_AFTER):
            return
        if now - self._last_idle < self.IDLE_TICK_S:
            return
        self._last_idle = now
        # ease the resting baseline back toward neutral (return-to-center when left alone)
        for k in ("yaw", "pitch", "body"):
            self._pose[k] *= (1.0 - self.IDLE_RECENTER)
        # layered slow sines at incommensurate periods -> organic, non-repetitive
        # sway, added ON TOP of (not stored in) the baseline so jog/center stay
        # authoritative.
        body = self._pose["body"] + self.IDLE_SWAY_DEG * np.sin(now / 11.0 * 2 * np.pi)
        pitch = self._pose["pitch"] + self.IDLE_PITCH_DEG * np.sin(now / 7.0 * 2 * np.pi + 1.0)
        yaw = self._pose["yaw"] + self.IDLE_YAW_DEG * np.sin(now / 17.0 * 2 * np.pi)
        from reachy_mini.utils import create_head_pose
        head = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
        try:
            with self._lock:
                self.mini.goto_target(head=head, body_yaw=np.deg2rad(body),
                                      duration=self.IDLE_TICK_S * 1.4)
        except Exception:
            pass

    def _detect_faces(self, frame):
        import os
        import cv2
        if self._cascade is None:
            self._cascade = cv2.CascadeClassifier(
                os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

    # Bounded face tracking: we aim the head ourselves (instead of the SDK's
    # unbounded look_at_image) so tracking can never command a pose where the
    # head rim contacts the shell -- the bump zone is deep pitch at yaw.
    FT_YAW_MAX_DEG = 38.0      # head-only; the follow layer brings the body for more
    FT_PITCH_UP_DEG = -18.0    # looking up (negative = up); shell clearance limit
    FT_PITCH_DOWN_DEG = 25.0   # looking down
    FT_GAIN_YAW = 16.0         # deg of correction for a face at the frame edge
    FT_GAIN_PITCH = 11.0

    def _face_track_tick(self):
        now = time.time()
        if now - self._last_face < 0.25:             # ~4 Hz tracking
            return False
        self._last_face = now
        try:
            frame = self.mini.media.get_frame()
        except Exception:
            return False
        if frame is None:
            return False
        faces = self._detect_faces(frame)
        if len(faces) == 0:
            return False
        self._face_seen_ts = now                             # eyes have a target
        self._active_ts = now                                # real activity -> hold off idle motion
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # largest face
        cx, cy = x + w // 2, y + h // 2
        H, W = frame.shape[:2]
        if abs(cx - W / 2) < W * 0.08 and abs(cy - H / 2) < H * 0.08:
            return True                              # already centered -> hold
        _, head_yaw, head_pitch = self._read_pose()
        if head_yaw is None:
            return False
        dx = (cx - W / 2) / (W / 2)                  # -1..1, + = face right of center
        dy = (cy - H / 2) / (H / 2)                  # + = face below center
        new_yaw = max(-self.FT_YAW_MAX_DEG, min(self.FT_YAW_MAX_DEG,
                      float(np.rad2deg(head_yaw)) - dx * self.FT_GAIN_YAW))
        new_pitch = max(self.FT_PITCH_UP_DEG, min(self.FT_PITCH_DOWN_DEG,
                        float(np.rad2deg(head_pitch)) + dy * self.FT_GAIN_PITCH))
        from reachy_mini.utils import create_head_pose
        pose = create_head_pose(yaw=new_yaw, pitch=new_pitch, degrees=True)
        with self._lock:
            try:
                self.mini.goto_target(head=pose, duration=0.3)
            except Exception:
                pass
        return True

    def _emotion_for(self, text):
        t = text.lower()
        rules = [
            (("thank",), "grateful1"),
            (("haha", "lol", "funny", "hilar"), "cheerful1"),
            (("wow", "whoa", "incredible", "amazing"), "amazed1"),
            (("sorry", "unfortunately", "afraid", "can't", "cannot"), "downcast1"),
            (("hmm", "not sure", "confus", "unclear"), "confused1"),
        ]
        for kws, emo in rules:
            if any(k in t for k in kws):
                return emo
        if "!" in text:
            return "enthusiastic1"
        if "?" in text:
            return "curious1"
        return None

    def _get_emotion_move(self, name):
        try:
            self._load_moves()
            return self.emotions.get(name)
        except Exception:
            return None

    # ---------- LLM action tags ----------
    @staticmethod
    def _extract_actions(text):
        """Pull [dance]/[gesture:NAME]/[look:DIR] tags out of a reply.
        Returns (actions, clean_text) where actions is [(kind, arg|None), ...]."""
        actions = [(m.group(1).lower(), (m.group(2) or "").strip().lower() or None)
                   for m in ACTION_RX.finditer(text or "")]
        clean = re.sub(r"\s{2,}", " ", ACTION_RX.sub("", text or "")).strip()
        return actions, clean

    def _run_actions(self, actions):
        for kind, arg in actions[:2]:            # at most two actions per reply
            try:
                if kind == "look":
                    yaw, pitch = LOOK_PRESETS.get(arg or "center", (0, 0))
                    threading.Thread(target=self.look, args=(yaw, pitch), daemon=True).start()
                elif kind == "gesture":
                    self._enqueue_move(self._load_gesture(arg)) if arg else None
                elif kind == "dance":
                    self._load_moves()
                    from reachy_mini_dances_library import DanceMove
                    name = arg if arg in (self._dances or []) else random.choice(self._dances)
                    self._enqueue_move(DanceMove(name))
            except Exception as e:
                self._log("system", f"action {kind}:{arg} failed: {e}")

    # ---------- learned gestures (hand-guided record + replay) ----------
    @staticmethod
    def _gesture_path(name):
        slug = re.sub(r"[^a-z0-9_\-]", "_", (name or "").strip().lower())[:40]
        return (os.path.join(GESTURE_DIR, slug + ".json"), slug) if slug else (None, None)

    def list_gestures(self):
        try:
            return sorted(f[:-5] for f in os.listdir(GESTURE_DIR) if f.endswith(".json"))
        except Exception:
            return []

    def _load_gesture(self, name):
        from reachy_mini.motion.recorded_move import RecordedMove
        path, _ = self._gesture_path(name)
        if not path or not os.path.exists(path):
            return None
        with open(path) as f:
            return RecordedMove(json.load(f))

    def gesture_record_start(self):
        if self.mini is None:
            return {"error": "robot not connected"}
        if self._recording_gesture:
            return {"error": "already recording"}
        self._recording_gesture = True           # behaviors + flutter stand down
        with self._lock:
            self.mini.disable_motors()           # limp -> move him by hand
            self.mini.start_recording()
        self._log("system", "gesture recording -- move him by hand")
        return {"ok": True, "recording": True}

    def gesture_record_stop(self, name, save=True):
        if self.mini is None or not self._recording_gesture:
            return {"error": "not recording"}
        with self._lock:
            data = self.mini.stop_recording()
            self.mini.enable_motors()            # pins targets to present pose (no snap)
        self._recording_gesture = False
        if not save:
            return {"ok": True, "discarded": True}
        if not data:
            return {"error": "nothing recorded"}
        path, slug = self._gesture_path(name)
        if not path:
            return {"error": f"bad gesture name {name!r}"}
        times = [float(r.get("time", i * 0.02)) for i, r in enumerate(data)]
        t0 = times[0]
        move = {"description": slug, "time": [t - t0 for t in times], "set_target_data": data}
        with open(path, "w") as f:
            json.dump(move, f, default=float)    # tolerate stray numpy scalars
        self._log("system", f"learned gesture '{slug}' ({move['time'][-1]:.1f}s, {len(data)} frames)")
        self._refresh_actions_hint()
        return {"ok": True, "name": slug, "seconds": round(move["time"][-1], 1)}

    def _refresh_actions_hint(self):
        from twin import brains as _brains
        _brains.set_actions_hint(self._dances or [], self.list_gestures())

    def gesture_play(self, name):
        if self.mini is None:
            return {"error": "robot not connected"}
        move = self._load_gesture(name)
        if move is None:
            return {"error": f"unknown gesture {name!r}"}
        self._enqueue_move(move)
        return {"ok": True, "name": name}

    def gesture_delete(self, name):
        path, slug = self._gesture_path(name)
        if path and os.path.exists(path):
            os.remove(path)
            self._refresh_actions_hint()
            return {"ok": True, "deleted": slug}
        return {"error": f"unknown gesture {name!r}"}

    # ---------- camera + manual jog ----------
    def get_jpeg(self, quality=70, timeout=0.5):
        # bound get_frame() so a stalled/empty camera pipeline can't hang the request
        result = [None]

        def _grab():
            try:
                result[0] = self.mini.media.get_frame()
            except Exception:
                pass

        t = threading.Thread(target=_grab, daemon=True)
        t.start()
        t.join(timeout)
        frame = result[0]
        if frame is None:
            return None
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    _JOG_LIMITS = {"pitch": 30, "roll": 30, "yaw": 60, "body": 90, "ant": 90}  # degrees
    ANT_REST_DEG = 8   # antennas held at exactly 0 deg hunt (backlash); a small offset stops it

    def _apply_pose(self):
        mini = self.mini                       # snapshot: the supervisor may null it mid-call
        if mini is None:
            return dict(self._pose)
        from reachy_mini.utils import create_head_pose
        p = self._pose
        head = create_head_pose(roll=p["roll"], pitch=p["pitch"], yaw=p["yaw"], degrees=True)
        ant = np.deg2rad([p["ant"] + self.ANT_REST_DEG, p["ant"] + self.ANT_REST_DEG])
        try:
            with self._lock:
                mini.goto_target(head=head, body_yaw=np.deg2rad(p["body"]),
                                 antennas=ant, duration=0.35)
        except Exception as e:                 # link dropped mid-move -> don't 500 the endpoint
            self._log("system", f"apply_pose failed (link?): {e}")
        return dict(self._pose)

    def servos(self):
        """Live motor telemetry for the panel. Non-blocking: if the robot lock is busy
        (e.g. he's mid-speech), return the last-known values instead of stalling the
        request until the utterance ends."""
        out = {"mode": self._cached_mode, **self._servo_cache}
        if self._lock.acquire(timeout=0.2):
            try:
                head, ant = self.mini.get_current_joint_positions()
                m = np.asarray(self.mini.get_current_head_pose(), dtype=float)
                self._servo_cache = {
                    "body_yaw": round(float(np.rad2deg(head[0])), 1),   # head[0] = body yaw
                    "head_yaw": round(float(np.rad2deg(np.arctan2(m[1, 0], m[0, 0]))), 1),
                    "head_pitch": round(float(np.rad2deg(-np.arcsin(max(-1.0, min(1.0, m[2, 0]))))), 1),
                    "antenna_left": round(float(np.rad2deg(ant[0])), 1),
                    "antenna_right": round(float(np.rad2deg(ant[1])), 1),
                }
                out.update(self._servo_cache)
            except Exception:
                pass
            finally:
                self._lock.release()
        now = time.time()
        if now - self._mode_ts > 10:
            try:
                self._cached_mode = json.loads(
                    urllib.request.urlopen(DAEMON + "/api/motors/status", timeout=3).read()).get("mode")
            except Exception:
                pass
            self._mode_ts = now
        out["mode"] = self._cached_mode
        return out

    # A manual jog/look/center suppresses the autonomous behaviors this long, so
    # turn-to-sound / face-track / follow don't immediately revert the human's move.
    MANUAL_HOLD_S = 4.0

    def _mark_manual(self, hold=None):
        """Tell the behavior loop a human is driving -- stand down until `hold` secs."""
        now = time.time()
        self._manual_until = now + (self.MANUAL_HOLD_S if hold is None else hold)
        self._active_ts = now              # real activity -> hold off idle motion

    def look(self, yaw_deg, pitch_deg):
        """Aim the head at an absolute yaw/pitch -- used by the iPad app's Vision
        face-tracking. Routed through _pose so look, jog, and center share one state."""
        y, p = float(yaw_deg), float(pitch_deg)
        if not (np.isfinite(y) and np.isfinite(p)):   # reject NaN/inf from a tracker glitch
            return dict(self._pose)
        self._pose["yaw"] = max(-self._JOG_LIMITS["yaw"], min(self._JOG_LIMITS["yaw"], y))
        self._pose["pitch"] = max(-self._JOG_LIMITS["pitch"], min(self._JOG_LIMITS["pitch"], p))
        # Short hold: the iPad streams look() continuously, so this just keeps the
        # hub's own behaviors from fighting the external tracker while it's active.
        self._mark_manual(hold=1.5)
        return self._apply_pose()

    def jog(self, part, delta):
        if part not in self._pose:
            return dict(self._pose)
        d = float(delta)
        if not np.isfinite(d):                # reject a bad delta rather than command NaN
            return dict(self._pose)
        # Re-sync from the servos for the axes the behavior layer moves behind
        # _pose's back, so a jog nudges from where he ACTUALLY is, not a stale target.
        b, y, p = self._read_pose()
        if part == "body" and b is not None:
            self._pose["body"] = float(np.rad2deg(b))
        elif part == "yaw" and y is not None:
            self._pose["yaw"] = float(np.rad2deg(y))
        elif part == "pitch" and p is not None:
            self._pose["pitch"] = float(np.rad2deg(p))
        lim = self._JOG_LIMITS[part]
        self._pose[part] = max(-lim, min(lim, self._pose[part] + d))
        self._mark_manual()
        return self._apply_pose()

    def center(self):
        for k in self._pose:
            self._pose[k] = 0.0
        self._doa_hist.clear()
        self._follow_since = None
        self._mark_manual()
        return self._apply_pose()

    # ---------- sleep / wake ----------
    # Sleep snapshots how he's set up + where he's posed, parks the behaviors,
    # stops listening, nods to a rest pose and relaxes the motors. Wake powers
    # back up and restores exactly that snapshot.
    _SLEEP_POSE = {"yaw": 0.0, "roll": 0.0, "pitch": 20.0, "body": 0.0, "ant": -14.0}
    SLEEP_ANIM = "sleep1"        # graceful "nodding off" emotion
    WAKE_ANIM = "welcoming1"     # cute perk-up greeting on wake (try cheerful1 / curious1 / surprised1)

    def sleep(self):
        if self._asleep:
            return {"asleep": True}
        # Remember how he was set up so wake() can put it all back.
        self._sleep_state = {
            "behaviors": dict(self.behaviors),
            "listening": self._listening,
            "brain": self.active,
            "pose": dict(self._pose),
        }
        for k in list(self.behaviors):          # stop all autonomous motion
            self.set_behavior(k, False)
        if self._listening:                     # stop the mic
            self.set_listening(False)
        self._asleep = True
        # Play the "going to sleep" emotion so his head settles gracefully into a
        # rest pose, and HOLD it (motors stay on) -- cutting the motors made the
        # head droop. Falls back to a manual rest pose if the library is missing.
        try:
            self._load_moves()
            with self._lock:
                self.mini.play_move(self.emotions.get(self.SLEEP_ANIM))
        except Exception as e:
            self._log("system", f"sleep anim unavailable: {e}")
        # The emotion ends back at neutral/alert, so settle him into a held rest
        # pose (head dipped, antennas down) -- slow + smooth -- so he actually
        # ENDS looking asleep. Motors stay on, so no droop.
        self._pose.update(self._SLEEP_POSE)
        try:
            from reachy_mini.utils import create_head_pose
            p = self._pose
            head = create_head_pose(roll=p["roll"], pitch=p["pitch"], yaw=p["yaw"], degrees=True)
            ant = np.deg2rad([p["ant"] + self.ANT_REST_DEG, p["ant"] + self.ANT_REST_DEG])
            with self._lock:
                self.mini.goto_target(head=head, body_yaw=np.deg2rad(p["body"]),
                                      antennas=ant, duration=1.5)
        except Exception:
            self._apply_pose()
        self._log("system", "💤 asleep")
        return {"asleep": True}

    def wake(self):
        if not self._asleep:
            return {"asleep": False}
        st = self._sleep_state or {}
        self._asleep = False
        # cute perk-up greeting, then settle back to how he was
        try:
            self._load_moves()
            with self._lock:
                self.mini.play_move(self.emotions.get(self.WAKE_ANIM))
        except Exception as e:
            self._log("system", f"wake anim unavailable: {e}")
        if "pose" in st:                         # smoothly back to where he was
            self._pose.update(st["pose"])
            self._apply_pose()
        if st.get("brain") in self.brains:
            self.active = st["brain"]
        for k, v in st.get("behaviors", {}).items():
            self.set_behavior(k, v)
        if st.get("listening"):
            self.set_listening(True)
        self._sleep_state = None
        self._mark_manual()                      # grace period before behaviors re-engage
        self._log("system", "☀️ awake")
        return {"asleep": False}

    # ---------- volume (proxy daemon) ----------
    def get_volume(self):
        try:
            return json.loads(urllib.request.urlopen(DAEMON + "/api/volume/current", timeout=5).read())
        except Exception as e:
            return {"error": str(e)}

    # only treat these as commands when they're unambiguous -- "what does mute mean?"
    # must reach the brain, not silence the robot.
    _MUTE_RX = re.compile(
        r"(?:hey\s+\w+[,\s]+)?(?:please\s+)?(?:mute|silence)"
        r"(?:\s+(?:yourself|it|the\s+(?:volume|sound|audio)))?\s*[.!]*", re.I)
    _VOL_NUM_RX = re.compile(r"\b(?:set|turn|put)?\s*(?:the\s+)?volume\s+(?:up\s+|down\s+)?(?:to|at)?\s*(\d{1,3})\b", re.I)
    _VOL_DOWN_RX = re.compile(r"\b(quieter|too loud|turn (?:it|the volume) down|turn down|volume down|softer|less loud|lower (?:the )?(?:volume|it))\b", re.I)
    # "can't hear" alone hijacks "I can't hear the TV" -- require it to be about him.
    _VOL_UP_RX = re.compile(r"\b(louder|speak up|turn (?:it|the volume) up|turn up|can'?t hear (?:you|ya|u)|volume up|more volume)\b", re.I)

    def _maybe_volume_command(self, text):
        """Intercept spoken/typed volume commands. Returns True if handled (no brain call)."""
        t = text.lower().strip()
        # Decide the command from the text FIRST; only consult the (cached, never
        # blocking) current volume for the relative up/down cases. This keeps a
        # slow/down volume daemon from adding latency to every normal chat turn.
        if self._MUTE_RX.fullmatch(t):
            target = 0
        elif "volume" in t and re.search(r"\b(max|full|loudest|all the way)\b", t):
            target = 100
        elif (m := self._VOL_NUM_RX.search(t)):
            target = int(m.group(1))
        elif self._VOL_DOWN_RX.search(t):
            cur = self._volume_cached()
            target = (cur if isinstance(cur, (int, float)) else 60) - 15
        elif self._VOL_UP_RX.search(t):
            cur = self._volume_cached()
            target = (cur if isinstance(cur, (int, float)) else 60) + 15
        else:
            return False
        target = max(0, min(100, target))
        self.set_volume(target)
        self._log("you", text)
        self._log("system", f"volume -> {target}")
        self.say(f"Okay, volume {target}." if target else "Muted.")
        return True

    def set_volume(self, v):
        self._vol_cache = int(v)            # reflect immediately in the panel
        self._vol_ts = time.time()
        body = json.dumps({"volume": int(v)}).encode()
        req = urllib.request.Request(DAEMON + "/api/volume/set", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            return json.loads(urllib.request.urlopen(req, timeout=5).read())
        except Exception as e:
            return {"error": str(e)}

    def _volume_cached(self):
        """Never blocks: refreshes in the background so /api/state stays fast even
        when the daemon is slow or down."""
        if time.time() - self._vol_ts > 3 and not self._vol_refreshing:
            self._vol_refreshing = True
            threading.Thread(target=self._refresh_volume, daemon=True).start()
        return self._vol_cache

    def _refresh_volume(self):
        try:
            v = self.get_volume().get("volume")
            if v is not None:
                self._vol_cache = int(v)
        finally:
            self._vol_ts = time.time()
            self._vol_refreshing = False

    # ---------- state for the panel ----------
    def state(self):
        return {
            "active": self.active,
            "listening": self._listening,
            "volume": self._volume_cached(),
            "brains": list(self.brains),
            "voices": self.voices,
            "behaviors": self.behaviors,
            "asleep": self._asleep,
            "robot": "connected" if self.robot_connected else "disconnected",
            "robot_error": None if self.robot_connected else self.last_error,
            "log": list(self.log)[-40:],
        }
