#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Union

import numpy as np
import sounddevice as sd

from .microwakeword import MicroWakeWord, MicroWakeWordFeatures
from .models import AvailableWakeWord, Preferences, ServerState, WakeWordType
from .mpv_player import MpvMediaPlayer
from .openwakeword import OpenWakeWord, OpenWakeWordFeatures
from .satellite import VoiceSatelliteProtocol
from .util import get_mac, is_arm
from .zeroconf import HomeAssistantZeroconf

_LOGGER = logging.getLogger(__name__)
_MODULE_DIR = Path(__file__).parent
_REPO_DIR = _MODULE_DIR.parent
_WAKEWORDS_DIR = _REPO_DIR / "wakewords"
_OWW_DIR = _WAKEWORDS_DIR / "openWakeWord"
_SOUNDS_DIR = _REPO_DIR / "sounds"

if is_arm():
    _LIB_DIR = _REPO_DIR / "lib" / "linux_arm64"
else:
    _LIB_DIR = _REPO_DIR / "lib" / "linux_amd64"


# -----------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--audio-input-device",
        default=None,
        help="Use default ALSA input (no need to specify manually)",
    )
    parser.add_argument(
        "--audio-input-block-size",
        type=int,
        default=1024,
        help="Audio block size for input stream",
    )
    parser.add_argument(
        "--audio-output-device",
        default=None,
        help="Use default ALSA output (no need to specify manually)",
    )
    parser.add_argument(
        "--wake-word-dir",
        default=[_WAKEWORDS_DIR],
        action="append",
        help="Directory with wake word models (.tflite) and configs (.json)",
    )
    parser.add_argument("--wake-model", default="okay_nabu", help="Id of active wake model")
    parser.add_argument("--stop-model", default="stop", help="Id of stop model")
    parser.add_argument("--refractory-seconds", default=2.0, type=float)
    parser.add_argument(
        "--oww-melspectrogram-model",
        default=_OWW_DIR / "melspectrogram.tflite",
    )
    parser.add_argument(
        "--oww-embedding-model",
        default=_OWW_DIR / "embedding_model.tflite",
    )
    parser.add_argument(
        "--wakeup-sound", default=str(_SOUNDS_DIR / "wake_word_triggered.flac")
    )
    parser.add_argument(
        "--timer-finished-sound", default=str(_SOUNDS_DIR / "timer_finished.flac")
    )
    parser.add_argument("--preferences-file", default=_REPO_DIR / "preferences.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6053)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.info("🎧 Using system default ALSA input/output (same as arecord/aplay)")

    # Load available wake words
    wake_word_dirs = [Path(ww_dir) for ww_dir in args.wake_word_dir]
    available_wake_words: Dict[str, AvailableWakeWord] = {}

    for wake_word_dir in wake_word_dirs:
        for model_config_path in wake_word_dir.glob("*.json"):
            model_id = model_config_path.stem
            if model_id == args.stop_model:
                continue
            with open(model_config_path, "r", encoding="utf-8") as model_config_file:
                model_config = json.load(model_config_file)
                available_wake_words[model_id] = AvailableWakeWord(
                    id=model_id,
                    type=WakeWordType(model_config["type"]),
                    wake_word=model_config["wake_word"],
                    trained_languages=model_config.get("trained_languages", []),
                    config_path=model_config_path,
                )

    # Load preferences
    preferences_path = Path(args.preferences_file)
    if preferences_path.exists():
        with open(preferences_path, "r", encoding="utf-8") as preferences_file:
            preferences = Preferences(**json.load(preferences_file))
    else:
        preferences = Preferences()

    libtensorflowlite_c_path = _LIB_DIR / "libtensorflowlite_c.so"

    # Load wake/stop models
    wake_models: Dict[str, Union[MicroWakeWord, OpenWakeWord]] = {}
    if preferences.active_wake_words:
        for wake_word_id in preferences.active_wake_words:
            wake_word = available_wake_words.get(wake_word_id)
            if wake_word is None:
                continue
            wake_models[wake_word_id] = wake_word.load(libtensorflowlite_c_path)
    if not wake_models:
        wake_word_id = args.wake_model
        wake_word = available_wake_words[wake_word_id]
        wake_models[wake_word_id] = wake_word.load(libtensorflowlite_c_path)

    # Load stop model
    stop_model: Optional[MicroWakeWord] = None
    for wake_word_dir in wake_word_dirs:
        stop_config_path = wake_word_dir / f"{args.stop_model}.json"
        if stop_config_path.exists():
            stop_model = MicroWakeWord.from_config(stop_config_path, libtensorflowlite_c_path)
            break
    assert stop_model is not None

    state = ServerState(
        name=args.name,
        mac_address=get_mac(),
        audio_queue=Queue(),
        entities=[],
        available_wake_words=available_wake_words,
        wake_words=wake_models,
        stop_word=stop_model,
        music_player=MpvMediaPlayer(device=args.audio_output_device or "alsa/default"),
        tts_player=MpvMediaPlayer(device=args.audio_output_device or "alsa/default"),
        wakeup_sound=args.wakeup_sound,
        timer_finished_sound=args.timer_finished_sound,
        preferences=preferences,
        preferences_path=preferences_path,
        libtensorflowlite_c_path=libtensorflowlite_c_path,
        oww_melspectrogram_path=Path(args.oww_melspectrogram_model),
        oww_embedding_path=Path(args.oww_embedding_model),
        refractory_seconds=args.refractory_seconds,
    )

    process_audio_thread = threading.Thread(
        target=process_audio, args=(state,), daemon=True
    )
    process_audio_thread.start()

    def sd_callback(indata, _frames, _time, _status):
        state.audio_queue.put_nowait(bytes(indata))

    loop = asyncio.get_running_loop()
    server = await loop.create_server(
        lambda: VoiceSatelliteProtocol(state), host=args.host, port=args.port
    )

    # Zeroconf discovery
    discovery = HomeAssistantZeroconf(port=args.port, name=args.name)
    await discovery.register_server()

    try:
        _LOGGER.info("🎙️ Opening default ALSA input (no device specified)")
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=args.audio_input_block_size,
            device=args.audio_input_device or None,
            dtype="int16",
            channels=1,
            callback=sd_callback,
        ):
            async with server:
                _LOGGER.info("Server started (host=%s, port=%s)", args.host, args.port)
                await server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.audio_queue.put_nowait(None)
        process_audio_thread.join()

    _LOGGER.info("Server stopped")


# -----------------------------------------------------------------------------
def process_audio(state: ServerState):
    wake_words: List[Union[MicroWakeWord, OpenWakeWord]] = []
    micro_features: Optional[MicroWakeWordFeatures] = None
    micro_inputs: List[np.ndarray] = []

    oww_features: Optional[OpenWakeWordFeatures] = None
    oww_inputs: List[np.ndarray] = []
    has_oww = False
    last_active: Optional[float] = None

    try:
        while True:
            audio_chunk = state.audio_queue.get()
            if audio_chunk is None:
                break

            if state.satellite is None:
                continue

            if (not wake_words) or (state.wake_words_changed and state.wake_words):
                state.wake_words_changed = False
                wake_words = [ww for ww in state.wake_words.values() if ww.is_active]
                has_oww = any(isinstance(ww, OpenWakeWord) for ww in wake_words)
                if micro_features is None:
                    micro_features = MicroWakeWordFeatures(
                        libtensorflowlite_c_path=state.libtensorflowlite_c_path,
                    )
                if has_oww and (oww_features is None):
                    oww_features = OpenWakeWordFeatures(
                        melspectrogram_model=state.oww_melspectrogram_path,
                        embedding_model=state.oww_embedding_path,
                        libtensorflowlite_c_path=state.libtensorflowlite_c_path,
                    )

            try:
                state.satellite.handle_audio(audio_chunk)
                micro_inputs.clear()
                micro_inputs.extend(micro_features.process_streaming(audio_chunk))

                if has_oww:
                    oww_inputs.clear()
                    oww_inputs.extend(oww_features.process_streaming(audio_chunk))

                for wake_word in wake_words:
                    activated = False
                    if isinstance(wake_word, MicroWakeWord):
                        for micro_input in micro_inputs:
                            if wake_word.process_streaming(micro_input):
                                activated = True
                    elif isinstance(wake_word, OpenWakeWord):
                        for oww_input in oww_inputs:
                            for prob in wake_word.process_streaming(oww_input):
                                if prob > 0.5:
                                    activated = True

                    if activated:
                        now = time.monotonic()
                        if (last_active is None) or ((now - last_active) > state.refractory_seconds):
                            state.satellite.wakeup(wake_word)
                            last_active = now

                stopped = False
                for micro_input in micro_inputs:
                    if state.stop_word.process_streaming(micro_input):
                        stopped = True
                if stopped and state.stop_word.is_active:
                    state.satellite.stop()
            except Exception:
                _LOGGER.exception("Unexpected error handling audio")

    except Exception:
        _LOGGER.exception("Unexpected error processing audio")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
