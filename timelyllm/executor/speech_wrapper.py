import os
import time
import wave
import shutil
import tempfile
import threading
import subprocess
import re
import numpy as np
import soundfile as sf
from typing import Optional, Dict, Tuple

# Piper imports (optional, will be imported only if using piper engine)
try:
    from piper import PiperVoice, SynthesisConfig
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException as ORTRuntimeException
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    PiperVoice = None
    SynthesisConfig = None
    ORTRuntimeException = Exception

# Piper text processing functions
_RE_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d]")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_RE_PATH = re.compile(r"(?<!\w)/(?:[\w\-.]+/)+[\w\-.]*")
_RE_EMOJI = re.compile(
    r"[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]"
)
_RE_MULTI_PUNCT = re.compile(r"([.,!?;:'\"()-])\1+")
_RE_SPACE = re.compile(r"\s+")
_RE_EN_ALLOWED = re.compile(r"[^A-Za-z0-9\s\.,!\?:;\'\"\-\(\)]")
_RE_SENT_SPLIT = re.compile(r'([.!?])')

def sanitize_text_en(text: str) -> str:
    """Clean text for piper TTS."""
    t = (text or "")
    t = _RE_ZERO_WIDTH.sub("", t)
    t = _RE_URL.sub(" link ", t)
    t = _RE_EMAIL.sub(" email ", t)
    t = _RE_PATH.sub(" path ", t)
    t = _RE_EMOJI.sub(" emoji ", t)
    t = (t.replace(""", "\"").replace(""", "\"")
           .replace("'", "'").replace("'", "'")
           .replace("—", "-").replace("–", "-").replace("_", " ").replace("/", " "))
    t = _RE_EN_ALLOWED.sub("", t)
    t = _RE_MULTI_PUNCT.sub(r"\1", t)
    t = _RE_SPACE.sub(" ", t).strip()
    return t

def split_sentences_en(text: str, max_chars: int = 160) -> list:
    """Split text into sentences for piper TTS."""
    if not text:
        return []
    parts, tokens = [], _RE_SENT_SPLIT.split(text)
    for i in range(0, len(tokens), 2):
        seg = tokens[i].strip()
        end = tokens[i+1] if i+1 < len(tokens) else ""
        if not seg:
            continue
        parts.append((seg + (end if end else "")).strip())

    chunks = []
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if not any(c.isalpha() for c in s):
            continue
        if len(s) <= max_chars:
            chunks.append(s)
        else:
            words, cur, cur_len = s.split(" "), [], 0
            for w in words:
                add = len(w) + (1 if cur else 0)
                if cur_len + add > max_chars:
                    if cur and any(c.isalpha() for c in cur[0]):
                        chunks.append(" ".join(cur))
                    cur, cur_len = [w], len(w)
                else:
                    cur.append(w); cur_len += add
            if cur and any(c.isalpha() for c in cur[0]):
                chunks.append(" ".join(cur))
    return chunks if chunks else ["This is a placeholder."]

class _WaveSink:
    """Persistent WAV append writer (thread-safe)."""
    def __init__(self, filename: str):
        self.filename = filename
        self._wf: Optional[wave.Wave_write] = None
        self._params: Optional[Tuple[int, int, int]] = None  # (nchan, sampwidth, framerate)
        self._lock = threading.Lock()

    def _open_with_params(self, nchan: int, sw: int, fr: int):
        os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
        self._wf = wave.open(self.filename, 'wb')
        self._wf.setnchannels(nchan)
        self._wf.setsampwidth(sw)
        self._wf.setframerate(fr)
        self._params = (nchan, sw, fr)

    def ensure_open(self, nchan: int, sw: int, fr: int):
        if self._wf is None:
            self._open_with_params(nchan, sw, fr)
        else:
            if self._params != (nchan, sw, fr):
                raise RuntimeError(f"WAV format mismatch: sink={self._params}, new={(nchan, sw, fr)}")

    def append_silence(self, seconds: float):
        if seconds <= 0:
            return
        if self._wf is None or self._params is None:
            raise RuntimeError("append_silence called before sink params established")
        nchan, sw, fr = self._params
        nframes = int(seconds * fr)
        # Write zero frames directly to avoid extra IO
        self._wf.writeframes(b"\x00" * nframes * nchan * sw)

    def append_from_wav(self, src_wav: str):
        with wave.open(src_wav, 'rb') as r:
            nchan, sw, fr = r.getnchannels(), r.getsampwidth(), r.getframerate()
            if self._wf is None:
                self._open_with_params(nchan, sw, fr)
            else:
                if (nchan, sw, fr) != self._params:
                    raise RuntimeError(f"WAV format mismatch: sink={self._params}, src={(nchan, sw, fr)}")
            # Stream-copy frames to avoid memory spikes
            while True:
                chunk = r.readframes(4096)
                if not chunk:
                    break
                self._wf.writeframes(chunk)

    def close(self):
        if self._wf is not None:
            self._wf.close()
            self._wf = None

class SpeechGenerator:
    """
    TTS generator supporting multiple engines:
    - piper: High-quality neural TTS (recommended)
    - espeak-ng: Lightweight CLI TTS
    - flite: Fallback CLI TTS
    - Thread-safe: same filename serialized by lock; different filenames can run in parallel.
    """
    def __init__(self, default_voice: Optional[str] = None, speed: Optional[int] = None, 
                 dry_run: bool = False, real_audio_task_ids=None, engine: str = "auto",
                 piper_model_path: Optional[str] = None):
        self.sr = None  # Determined by the first generated segment
        self._sinks: Dict[str, _WaveSink] = {}
        self._sinks_lock = threading.Lock()  # Protects _sinks dict
        self.dry_run = dry_run  # If True, no real WAV files are generated, only simulating delays

        # real_audio_task_ids: specifies which task_ids need real audio generation
        # Can be a single task_id (int) or a list (list), or None (use global dry_run setting)
        if real_audio_task_ids is not None:
            if isinstance(real_audio_task_ids, int):
                self.real_audio_task_ids = {real_audio_task_ids}
            elif isinstance(real_audio_task_ids, (list, set)):
                self.real_audio_task_ids = set(real_audio_task_ids)
            else:
                self.real_audio_task_ids = set()
        else:
            self.real_audio_task_ids = None
        
        # Piper specific settings
        self.piper_voice = None
        self.piper_config = None
        self.max_chars = 160
        
        # Select engine
        if not dry_run or self.real_audio_task_ids is not None:
            self.engine, self.cmd_builder = self._pick_engine(default_voice, speed, engine, piper_model_path)
        else:
            self.engine, self.cmd_builder = "dry_run", None

    # ---------- Engine selection and command building ----------
    def _pick_engine(self, default_voice: Optional[str], speed: Optional[int], 
                     engine: str = "auto", piper_model_path: Optional[str] = None):
        import shutil as _sh
        
        # If piper is specified or available in auto mode
        if engine == "piper" or (engine == "auto" and PIPER_AVAILABLE):
            if not PIPER_AVAILABLE:
                print("[WARN] Piper not available, falling back to espeak-ng/flite")
            else:
                try:
                    # Set default model path
                    if piper_model_path is None:
                        piper_model_path = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), 
                            "en_US-lessac-medium.onnx"
                        )
                    
                    if not os.path.exists(piper_model_path):
                        print(f"[WARN] Piper model not found at {piper_model_path}, falling back to espeak-ng/flite")
                    else:
                        # Initialize Piper
                        self.piper_voice = PiperVoice.load(piper_model_path)
                        self.piper_config = SynthesisConfig(noise_scale=0.667, length_scale=1.0, noise_w_scale=0.8)
                        self.piper_voice.config.sample_rate = 22050
                        print(f"[INFO] Using Piper TTS engine with model: {piper_model_path}")
                        return "piper", None
                except Exception as e:
                    print(f"[WARN] Failed to initialize Piper: {e}, falling back to espeak-ng/flite")
        
        # Fall back to espeak-ng or flite
        if _sh.which("espeak-ng"):
            def _cmd(text: str, out_wav: str):
                voice = default_voice or "en-us"
                sp = str(speed or 170)
                # Generate to temp file to avoid partial writes
                return ["espeak-ng", "-v", voice, "-s", sp, "-w", out_wav, text]
            print("[INFO] Using espeak-ng TTS engine")
            return "espeak-ng", _cmd
        elif _sh.which("flite"):
            def _cmd(text: str, out_wav: str):
                voice = default_voice or "slt"  # Common flite English voice
                # flite syntax: flite -voice slt "text" out.wav
                return ["flite", "-voice", voice, text, out_wav]
            print("[INFO] Using flite TTS engine")
            return "flite", _cmd
        else:
            raise RuntimeError("No TTS engine found. Please install 'piper-tts', 'espeak-ng' or 'flite'.")

    # ---------- Internal utilities ----------
    def _get_sink(self, filename: str) -> _WaveSink:
        with self._sinks_lock:
            sink = self._sinks.get(filename)
            if sink is None:
                sink = _WaveSink(filename)
                self._sinks[filename] = sink
            return sink

    def _synthesize_to_temp(self, text: str) -> str:
        # Generate temporary wav file
        tmpdir = tempfile.mkdtemp(prefix="tts_cli_")
        tmpwav = os.path.join(tmpdir, "seg.wav")
        cmd = self.cmd_builder(text, tmpwav)
        # Note: shell=False on Windows; Linux/macOS safe by default
        subprocess.run(cmd, check=True)
        return tmpwav  # Caller is responsible for cleaning up tmpdir
    
    def _synthesize_to_temp_piper(self, text: str) -> tuple:
        """Synthesize to temp file using Piper, returns (path, sr, sw, ch, nframes)"""
        tmp_path = tempfile.mktemp(suffix=".wav")
        try:
            with wave.open(tmp_path, "wb") as wf:
                self.piper_voice.synthesize_wav(text, wf, syn_config=self.piper_config)
                # If no frames were written, write a minimal silent frame
                if wf.getnframes() == 0:
                    wf.setframerate(self.piper_voice.config.sample_rate)
                    wf.setsampwidth(2)  # int16
                    wf.setnchannels(1)  # mono
                    wf.writeframes(b'\x00\x00')
            
            with wave.open(tmp_path, "rb") as rf:
                sr = rf.getframerate()
                sw = rf.getsampwidth()
                ch = rf.getnchannels()
                nframes = rf.getnframes()
            return tmp_path, sr, sw, ch, nframes
        except Exception:
            try: 
                os.remove(tmp_path)
            except: 
                pass
            raise
    
    def _append_from_wav_file_piper(self, src_path: str, dst_path: str, sr: int, sw: int, ch: int) -> float:
        """Append WAV file using soundfile (for Piper)"""
        if not os.path.exists(dst_path):
            shutil.copyfile(src_path, dst_path)
            with wave.open(src_path, "rb") as rf:
                nframes = rf.getnframes()
            return float(nframes) / sr
        else:
            total_frames = 0
            with sf.SoundFile(src_path, 'r') as sf_in:
                with sf.SoundFile(dst_path, 'r+') as sf_out:
                    assert sf_in.samplerate == sf_out.samplerate
                    assert sf_in.channels == sf_out.channels
                    sf_out.seek(0, sf.SEEK_END)
                    while True:
                        data = sf_in.read(48000, dtype='float32')
                        if data.shape[0] == 0:
                            break
                        sf_out.write(data)
                        total_frames += data.shape[0]
            return float(total_frames) / sr
    
    def _append_silence_piper(self, filename: str, sr: int, sw: int, ch: int, duration: float) -> float:
        """Append silence (for Piper)"""
        num_frames = int(sr * duration)
        if num_frames <= 0:
            return 0.0
        
        silence_pcm_i16 = np.zeros(num_frames, dtype=np.int16)
        silence_pcm_f32 = silence_pcm_i16.astype(np.float32) / 32768.0
        
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        if not os.path.exists(filename):
            with wave.open(filename, "wb") as wf:
                wf.setframerate(sr)
                wf.setsampwidth(sw)
                wf.setnchannels(ch)
                wf.writeframes(silence_pcm_i16.tobytes())
        else:
            with sf.SoundFile(filename, 'r+') as sf_out:
                sf_out.seek(0, sf.SEEK_END)
                sf_out.write(silence_pcm_f32)
        return duration
    
    def _speech_generate_piper(self, text: str, filename: str, silence_duration: float, task_id, t0: float) -> float:
        """Generate speech using Piper engine"""
        # Clean and split into sentences
        clean_all = sanitize_text_en(text)
        chunks = split_sentences_en(clean_all, max_chars=self.max_chars)
        
        if not chunks:
            chunks = ["..."]
        
        total_duration = 0.0
        pending_silence = float(silence_duration)
        
        for idx, chunk in enumerate(chunks, 1):
            c = sanitize_text_en(chunk)
            
            # Pre-check phonemization
            try:
                if not self.piper_voice.phonemize(c) or not self.piper_voice.phonemes_to_ids(self.piper_voice.phonemize(c)[0]):
                    print(f"[warn] replacing problematic chunk '{c}' with '...' due to empty phoneme IDs")
                    c = "..."
            except Exception as e:
                print(f"[warn] phonemization pre-check failed for '{c}': {e}, replacing with '...'")
                c = "..."
            
            if not c or c == ".":
                c = "..."
            
            try:
                tmp_path, sr, sw, ch, nframes = self._synthesize_to_temp_piper(c)
            except ORTRuntimeException as e:
                print(f"[warn] skipped a chunk due to ORT error: {e}")
                continue
            
            try:
                # Insert silence (only before the first segment)
                if pending_silence > 0:
                    total_duration += self._append_silence_piper(filename, sr, sw, ch, pending_silence)
                    pending_silence = 0
                
                total_duration += self._append_from_wav_file_piper(tmp_path, filename, sr, sw, ch)
                os.remove(tmp_path)
            except Exception as e:
                print(f"[error] failed to append to final file: {e}")
                raise
        
        elapsed = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] [REAL AUDIO - PIPER] task_id: {task_id}, generated ~{total_duration:.2f}s audio (elapsed {elapsed:.2f}s) to {filename}")
        return max(total_duration - elapsed, 0.0)

    # ---------- Public main interface ----------
    def speech_generate(self, text: str, filename: str, silence_duration: float = 0.0, task_id=None) -> float:
        """
        Generate speech and append to filename.
        Returns a rough delay (audio duration - synthesis time) for alignment estimation.

        If dry_run=True, no real files are generated, only estimates delays.
        If real_audio_task_ids is specified, only task_ids in that list generate real audio.
        """
        t0 = time.time()
        
        # Determine whether to generate real audio
        should_generate_real_audio = True
        if self.real_audio_task_ids is not None:
            # If real_audio_task_ids is specified, only task_ids in the list generate real audio
            should_generate_real_audio = task_id in self.real_audio_task_ids
        elif self.dry_run:
            # If real_audio_task_ids is not specified, use global dry_run setting
            should_generate_real_audio = False
        
        if not should_generate_real_audio:
            # Dry run mode: no real files generated, only estimate delays
            # Assume average speaking rate (adjust based on actual TTS)
            words = len(text.split())
            estimated_audio_duration = words * 0.3  # Assume 0.3s per word
            total_duration = estimated_audio_duration + silence_duration
            elapsed = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] [DRY RUN] task_id: {task_id}, text: {words} words, audio: ~{estimated_audio_duration:.2f}s, silence: {silence_duration:.2f}s, total: {total_duration:.2f}s -> {filename}")
            return max(estimated_audio_duration - elapsed, 0.0)
        
        # Normal mode: generate real WAV files
        # If using Piper engine
        if self.engine == "piper":
            return self._speech_generate_piper(text, filename, silence_duration, task_id, t0)
        
        # Use CLI engine (espeak-ng/flite)
        tmpwav = self._synthesize_to_temp(text)

        # Read segment params to establish/validate sink
        with wave.open(tmpwav, 'rb') as r:
            seg_frames = r.getnframes()
            fr = r.getframerate()

        sink = self._get_sink(filename)
        with sink._lock:
            # --- Correct order (silence first) ---
            # 1) Initialize sink with segment params (without writing frames)
            with wave.open(tmpwav, 'rb') as r:
                nchan, sw, fr = r.getnchannels(), r.getsampwidth(), r.getframerate()
                if sink._wf is None:
                    sink.ensure_open(nchan, sw, fr)
                else:
                    if sink._params != (nchan, sw, fr):
                        raise RuntimeError(f"WAV format mismatch: sink={sink._params}, seg={(nchan, sw, fr)}")

            # 2) Write silence
            if silence_duration > 0:
                sink.append_silence(silence_duration)

            # 3) Append segment
            sink.append_from_wav(tmpwav)

        # Clean up temporary directory
        tmpdir = os.path.dirname(tmpwav)
        try:
            os.remove(tmpwav)
            os.rmdir(tmpdir)
        except Exception:
            pass

        # Estimate audio duration and delay
        audio_duration = seg_frames / float(fr)
        elapsed = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] [REAL AUDIO] task_id: {task_id}, generated ~{audio_duration:.2f}s audio (elapsed {elapsed:.2f}s) to {filename}")
        return max(audio_duration - elapsed, 0.0)

    def close(self):
        # Close all open WAV files
        with self._sinks_lock:
            for sink in self._sinks.values():
                sink.close()
            self._sinks.clear()


if __name__ == "__main__":
    # text = "Hello, how are you?"
    # filename = "test.wav"
    # silence_duration = 0
    # speech_generator = SpeechGenerator()
    # delay_time = speech_generator.speech_generate(text, filename, silence_duration)
    # delay_time = speech_generator.speech_generate(text, filename, silence_duration)
    # print(f"delay_time: {delay_time}")

    # tts = SpeechGenerator()
    # fn = "test.wav"
    # tts.speech_generate("Hello, how are you?", fn, silence_duration=0.0)
    # tts.speech_generate("Nice to meet you!", fn, silence_duration=0.0)

    gen = SpeechGenerator()
    delay = gen.speech_generate("Hello, how are you?", "test.wav", silence_duration=1.0)
    delay = gen.speech_generate("Hello, how are you?", "test.wav", silence_duration=1.0)
    print("delay_time:", delay)