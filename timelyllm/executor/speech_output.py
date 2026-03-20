import os, re, time, tempfile, wave
import threading
import numpy as np
import soundfile as sf
from piper import PiperVoice, SynthesisConfig
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException as ORTRuntimeException
import shutil

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "en_US-lessac-medium.onnx")

_RE_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d]")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_RE_PATH = re.compile(r"(?<!\w)/(?:[\w\-.]+/)+[\w\-.]*")
_RE_EMOJI = re.compile(
    r"[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]"
)
_RE_MULTI_PUNCT = re.compile(r"([.,!?;:'\"()-])\1+")
_RE_SPACE = re.compile(r"\s+")
# Keep only: letters/digits/spaces/common English punctuation
_RE_EN_ALLOWED = re.compile(r"[^A-Za-z0-9\s\.,!\?:;\'\"\-\(\)]")
# Split by period/question mark/exclamation mark (preserve sentence-ending punctuation)
_RE_SENT_SPLIT = re.compile(r'([.!?])')

def sanitize_text_en(text: str) -> str:
    t = (text or "")
    t = _RE_ZERO_WIDTH.sub("", t)
    t = _RE_URL.sub(" link ", t)
    t = _RE_EMAIL.sub(" email ", t)
    t = _RE_PATH.sub(" path ", t)
    t = _RE_EMOJI.sub(" emoji ", t)
    t = (t.replace("“", "\"").replace("”", "\"")
           .replace("‘", "'").replace("’", "'")
           .replace("—", "-").replace("–", "-").replace("_", " ").replace("/", " "))
    t = _RE_EN_ALLOWED.sub("", t)
    t = _RE_MULTI_PUNCT.sub(r"\1", t)
    t = _RE_SPACE.sub(" ", t).strip()
    return t

def split_sentences_en(text: str, max_chars: int = 160) -> list[str]:
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
        # Skip pure punctuation/non-alphabetic segments (avoid empty phonemes)
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
    # Fallback: if everything was filtered out, provide a safe placeholder
    return chunks if chunks else ["This is a placeholder."]

class SpeechGenerator:
    def __init__(self, model_path: str = MODEL_PATH, volume: float = 0.5):
        self.voice = PiperVoice.load(model_path)
        self.cfg = SynthesisConfig(noise_scale=0.667, length_scale=1.0, noise_w_scale=0.8)
        self.voice.config.sample_rate = 22050
        self.max_chars = 160 # Max chars per chunk
        self.volume = max(0.1, min(1.0, volume))  # Clamp between 0.1 and 1.0
        self.lock = threading.Lock()

    def _append_pcm_f32(self, path: str, pcm_f32: np.ndarray, sr: int):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        pcm_f32 = np.asarray(pcm_f32, dtype=np.float32, order="C")
        pcm_f32 = pcm_f32 * self.volume  # Apply volume scaling
        if not os.path.exists(path):
            sf.write(path, np.zeros(0, dtype=np.float32), sr)
        with sf.SoundFile(path, mode="r+") as f:
            f.seek(0, sf.SEEK_END)
            f.write(pcm_f32)

    def _synthesize_to_tempfile(self, text: str):
        tmp_path = tempfile.mktemp(suffix=".wav")
        try:
            # Piper requires a wave.Wave_write object
            with wave.open(tmp_path, "wb") as wf:
                print(f"[debug] synthesizing to temp file {tmp_path} with {text}")
                self.voice.synthesize_wav(text, wf, syn_config=self.cfg)
                # If no frames were written, write a minimal silent frame to ensure header is valid
                if wf.getnframes() == 0:
                    wf.setframerate(self.voice.config.sample_rate)
                    wf.setsampwidth(2) # int16
                    wf.setnchannels(1) # mono
                    wf.writeframes(b'\x00\x00') # one silent frame
            with wave.open(tmp_path, "rb") as rf:
                sr = rf.getframerate()
                sw = rf.getsampwidth()
                ch = rf.getnchannels()
                nframes = rf.getnframes()
            return tmp_path, sr, sw, ch, nframes
        except Exception:
            try: os.remove(tmp_path)
            except: pass
            raise

    def _append_from_wav_file_in_chunks(self, src_path: str, dst_path: str, sr: int, sw: int, ch: int, chunk_frames: int = 48000):
        if sw != 2:
            raise RuntimeError(f"Unsupported sample width: {sw} bytes (expected 2 for int16).")
        with wave.open(src_path, "rb") as rf:
            assert rf.getframerate() == sr and rf.getsampwidth() == sw and rf.getnchannels() == ch
            while True:
                frames = rf.readframes(chunk_frames)
                if not frames:
                    break
                pcm_i16 = np.frombuffer(frames, dtype="<i2")
                if ch > 1:
                    pcm_i16 = pcm_i16.reshape(-1, ch)[:, 0]
                pcm_f32 = (pcm_i16.astype(np.float32) / 32768.0)
                self._append_pcm_f32(dst_path, pcm_f32, sr)

    def speech_generate(self, text: str, filename: str, silence_duration: float = 0.0) -> float:
        with self.lock:
            # 1) Clean + split sentences + limit length
            clean_all = sanitize_text_en(text)
            chunks = split_sentences_en(clean_all, max_chars=self.max_chars)

            # Ensure at least one segment
            if not chunks:
                chunks = ["..."]

            t0 = time.time()
            total_duration = 0.0

            # Prepend silence
            # Note: sr is only known after the first segment is synthesized; use a placeholder strategy here
            pending_silence = float(silence_duration)

            for idx, chunk in enumerate(chunks, 1):
                c = sanitize_text_en(chunk)  # Extra sanitization pass

                # Pre-check if phonemization results in empty phoneme IDs
                try:
                    if not self.voice.phonemize(c) or not self.voice.phonemes_to_ids(self.voice.phonemize(c)[0]):
                        print(f"[warn] replacing problematic chunk '{c}' with '...' due to empty phoneme IDs")
                        c = "..."
                except Exception as e:
                    print(f"[warn] phonemization pre-check failed for '{c}': {e}, replacing with '...'")
                    c = "..."

                if not c or c == ".":
                    # If empty or just a dot after cleaning, replace with safe placeholder
                    c = "..."

                try:
                    print(f"[debug] synthesizing to temp file {c}")
                    tmp_path, sr, sw, ch, nframes = self._synthesize_to_tempfile(c)
                    print(f"[debug] synthesized to temp file {tmp_path} with {sr, sw, ch, nframes}")
                except ORTRuntimeException as e:
                    # Piper/ORT error: segment was cleaned to empty or invalid, skip it
                    print(f"[warn] skipped a chunk due to ORT error: {e}")
                    continue

                try:
                    # Insert silence (only before the first segment)
                    if pending_silence > 0:
                        # Fill in silence
                        # print(f"[debug] inserting silence {pending_silence} for {c}")
                        total_duration += self._append_silence(filename, sr, sw, ch, pending_silence)
                        pending_silence = 0
          
                    total_duration += self._append_from_wav_file(tmp_path, filename, sr, sw, ch, nframes)
                    os.remove(tmp_path) # Delete after use to avoid accumulation
                except Exception as e:
                    print(f"[error] failed to append to final file: {e}")
                    raise

            return total_duration

    def _append_from_wav_file(self, src_path: str, dst_path: str, sr: int, sw: int, ch: int, nframes: int) -> float:
        # Read with soundfile and apply volume scaling
        total_frames = 0
        with sf.SoundFile(src_path, 'r') as sf_in:
            if not os.path.exists(dst_path):
                # Create new file
                with sf.SoundFile(dst_path, 'w', samplerate=sf_in.samplerate,
                                 channels=sf_in.channels, subtype=sf_in.subtype) as sf_out:
                    while True:
                        data = sf_in.read(48000, dtype='float32')
                        if not data.shape[0]:
                            break
                        data = data * self.volume  # Apply volume scaling
                        sf_out.write(data)
                        total_frames += data.shape[0]
            else:
                # Append to existing file
                with sf.SoundFile(dst_path, 'r+') as sf_out:
                    assert sf_in.samplerate == sf_out.samplerate
                    assert sf_in.channels == sf_out.channels
                    assert sf_in.subtype == sf_out.subtype

                    sf_out.seek(0, sf.SEEK_END)
                    while True:
                        data = sf_in.read(48000, dtype='float32')
                        if not data.shape[0]:
                            break
                        data = data * self.volume  # Apply volume scaling
                        sf_out.write(data)
                        total_frames += data.shape[0]
        return float(total_frames) / sr

    def _append_silence(self, filename: str, sr: int, sw: int, ch: int, duration: float) -> float:
        num_frames = int(sr * duration)
        if num_frames <= 0:
            return 0.0

        silence_pcm_i16 = np.zeros(num_frames, dtype=np.int16) # Silence is zeros
        silence_pcm_f32 = (silence_pcm_i16.astype(np.float32) / 32768.0)

        if not os.path.exists(filename):
            # Create new file, write silence
            with wave.open(filename, "wb") as wf:
                wf.setframerate(sr)
                wf.setsampwidth(sw)
                wf.setnchannels(ch)
                wf.writeframes(silence_pcm_i16.tobytes())
        else:
            # Append to existing file
            with sf.SoundFile(filename, 'r+') as sf_out:
                sf_out.seek(0, sf.SEEK_END)
                sf_out.write(silence_pcm_f32)
        return duration


if __name__ == "__main__":
    # text = "Hello, how are you?"
    # filename = "test.wav"
    # silence_duration = 0
    # speech_generator = SpeechGenerator()
    # delay_time = speech_generator.speech_generate(text, filename, silence_duration)
    # delay_time = speech_generator.speech_generate(text, filename, silence_duration)
    # print(f"delay_time: {delay_time}")

    tts = SpeechGenerator()
    fn = "test.wav"
    tts.speech_generate("The patient reports chest discomfort. What should I note in the record?", fn, silence_duration=0.0)
    # tts.speech_generate("Nice to meet you!", fn, silence_duration=0.0)