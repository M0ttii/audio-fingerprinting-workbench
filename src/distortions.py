"""
Audio distortion functions for the Audio Fingerprinting Benchmark.

Each function corresponds to one mutation type used in the query-generation
pipeline (NB 01). All functions operate on float32 numpy arrays in [-1, 1]
at a fixed sample rate (default 8 000 Hz, mono).

Design rules:
  - apply_speed_change MUST NOT normalise output length back to 10 s.
    The actual output duration is the correct result and must be stored
    in the manifest as duration_sec.
  - Zero-padding instead of exceptions for audio shorter than the window.
  - Seed-based determinism for all random choices.
  - apply_noise measures and returns the actual achieved SNR as a float
    alongside the mixed audio, so the notebook can log it.
"""

import io
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_and_resample(filepath: str, sr: int = 8000) -> np.ndarray:
    """Load an audio file, convert to mono, and resample to sr Hz.

    Uses librosa for decoding (supports MP3, WAV, FLAC, …).
    Output is a float32 array normalised to [-1, 1].

    Args:
        filepath: Path to the audio file.
        sr:       Target sample rate in Hz. Default: 8 000 Hz (Wang 2003,
                  Chang et al. 2021).

    Returns:
        1-D float32 numpy array of the audio signal at the given sample rate.

    Raises:
        FileNotFoundError: If filepath does not exist.
    """
    filepath = str(filepath)
    audio, _ = librosa.load(filepath, sr=sr, mono=True, dtype=np.float32)
    return audio


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def extract_segment(
    audio: np.ndarray,
    sr: int,
    duration_sec: float = 10.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Extract a random fixed-length segment from an audio array.

    The start position is chosen uniformly at random from the range
    [0, len(audio) - duration_samples]. This is reproducible via seed.

    If the audio is shorter than duration_sec, the signal is zero-padded
    to the requested length.

    Args:
        audio:        1-D float32 audio array.
        sr:           Sample rate of audio in Hz.
        duration_sec: Desired segment length in seconds. Default: 10.0.
        seed:         Random seed for reproducible start-point selection.
                      Pass track_id for per-track reproducibility.

    Returns:
        1-D float32 array of exactly int(duration_sec * sr) samples.
    """
    n_out = int(duration_sec * sr)

    if len(audio) <= n_out:
        # Zero-pad if signal is too short — never raise
        logger.warning(
            "Audio (%d samples) shorter than requested segment (%d samples). "
            "Zero-padding.", len(audio), n_out,
        )
        padded = np.zeros(n_out, dtype=np.float32)
        padded[:len(audio)] = audio
        return padded

    rng = np.random.default_rng(seed)
    max_start = len(audio) - n_out
    start = int(rng.integers(0, max_start + 1))
    return audio[start:start + n_out].copy()


# ---------------------------------------------------------------------------
# Tempo change (time-stretch, pitch-preserving)
# ---------------------------------------------------------------------------

def apply_tempo_change(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Time-stretch audio by rate while keeping pitch constant.

    Uses the phase-vocoder via librosa.effects.time_stretch.
    A rate > 1 makes the audio faster (shorter output).
    A rate < 1 makes the audio slower (longer output).

    Output length: len(audio) / rate  (approximately).
    Pitch: unchanged.

    Reference: Testkonzept Phase 1 — Gruppe A conditions A_tempo_*.

    Args:
        audio: 1-D float32 audio array.
        sr:    Sample rate (unused by time_stretch but kept for API symmetry).
        rate:  Speed-up factor. E.g. 0.95 = 5% slower, 1.20 = 20% faster.

    Returns:
        Time-stretched 1-D float32 array.
    """
    # librosa >= 0.9: keyword argument is 'rate'
    stretched = librosa.effects.time_stretch(y=audio, rate=rate)
    return stretched.astype(np.float32)


# ---------------------------------------------------------------------------
# Pitch shift (length-preserving)
# ---------------------------------------------------------------------------

def apply_pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by semitones without changing duration.

    Uses librosa.effects.pitch_shift (phase-vocoder + resampling).
    Output length equals input length. Tempo is unchanged.

    Reference: Testkonzept Phase 1 — Gruppe A conditions A_pitch_*.

    Args:
        audio:    1-D float32 audio array.
        sr:       Sample rate in Hz.
        semitones: Pitch shift in semitones. Positive = up, negative = down.
                  E.g. +1 = one semitone up (~5.9%), -2 = two semitones down.

    Returns:
        Pitch-shifted 1-D float32 array (same length as input).
    """
    shifted = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=semitones)
    return shifted.astype(np.float32)


# ---------------------------------------------------------------------------
# Speed change (tempo + pitch coupled, length changes)
# ---------------------------------------------------------------------------

def apply_speed_change(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Change playback speed by rate, altering both tempo and pitch.

    Implemented as sample-rate resampling (scipy.signal.resample).
    Unlike time-stretch, both tempo and pitch shift together — this mimics
    tape-speed change or vinyl pitch-bending.

    Output length: int(len(audio) / rate)  — NOT normalised back to 10 s.
    This is intentional: downstream pipelines must handle variable lengths.
    The actual duration must be stored in the manifest as duration_sec.

    Examples:
        rate=1.20: 10 s input → 8.33 s output  (faster + higher pitch)
        rate=0.80: 10 s input → 12.5 s output  (slower + lower pitch)

    Reference: Testkonzept Phase 1 — Gruppe A conditions A_speed_*.
               Fix #4: do NOT normalise output length.

    Args:
        audio: 1-D float32 audio array.
        sr:    Sample rate in Hz (unused here, kept for API symmetry).
        rate:  Speed factor. >1 = faster, shorter. <1 = slower, longer.

    Returns:
        Resampled 1-D float32 array with length int(len(audio) / rate).
    """
    n_out = int(len(audio) / rate)
    resampled = resample(audio, n_out)
    return resampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Noise addition (MUSAN)
# ---------------------------------------------------------------------------

def apply_noise(
    audio: np.ndarray,
    sr: int,
    noise_files: List[str],
    snr_db: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Mix a random noise file into audio at the requested SNR.

    A random file is selected from noise_files, loaded and resampled to sr,
    tiled/truncated to match the length of audio, and scaled using the
    standard SNR formula:

        P_signal = mean(audio ** 2)
        P_noise  = mean(noise ** 2)
        noise_scaled = noise * sqrt(P_signal / (P_noise * 10^(snr_db / 10)))

    The mixed signal is clipped to [-1, 1]. The actual achieved SNR is
    measured on the scaled noise before clipping and returned alongside
    the mixed audio, so the notebook can verify correctness.

    Args:
        audio:       1-D float32 signal array.
        sr:          Sample rate in Hz.
        noise_files: List of absolute paths to MUSAN .wav files (eval split).
        snr_db:      Target signal-to-noise ratio in dB.
        seed:        Random seed for noise-file selection.

    Returns:
        Tuple of:
            mixed  (np.ndarray): Mixed float32 audio, clipped to [-1, 1].
            actual_snr (float): Achieved SNR in dB, measured after scaling.

    Raises:
        ValueError: If noise_files is empty.
    """
    if not noise_files:
        raise ValueError("noise_files list is empty — cannot apply noise.")

    rng = random.Random(seed)
    noise_path = rng.choice(noise_files)

    noise_raw, _ = librosa.load(str(noise_path), sr=sr, mono=True,
                                dtype=np.float32)

    # Tile noise to cover the signal length, then trim
    n_sig = len(audio)
    if len(noise_raw) == 0:
        logger.warning("Empty noise file: %s. Returning unmodified audio.", noise_path)
        return audio.copy(), float("inf")

    n_repeats = int(np.ceil(n_sig / len(noise_raw)))
    noise_tiled = np.tile(noise_raw, n_repeats)[:n_sig]

    # Power-based SNR scaling
    # P_signal = mean(audio^2), P_noise = mean(noise^2)
    p_signal = float(np.mean(audio ** 2))
    p_noise = float(np.mean(noise_tiled ** 2))

    if p_noise < 1e-10:
        logger.warning("Near-silent noise file: %s. Returning unmodified audio.", noise_path)
        return audio.copy(), float("inf")

    if p_signal < 1e-10:
        logger.warning("Near-silent signal. SNR undefined — skipping noise.")
        return audio.copy(), float("inf")

    # noise_scaled = noise * sqrt(P_sig / (P_noise * 10^(snr/10)))
    scale = np.sqrt(p_signal / (p_noise * (10.0 ** (snr_db / 10.0))))
    noise_scaled = noise_tiled * scale

    # Measure actual SNR after scaling (before clipping)
    p_noise_scaled = float(np.mean(noise_scaled ** 2))
    actual_snr = 10.0 * np.log10(p_signal / p_noise_scaled) if p_noise_scaled > 0 else float("inf")

    mixed = np.clip(audio + noise_scaled, -1.0, 1.0).astype(np.float32)

    logger.debug(
        "apply_noise: target=%.1f dB, actual=%.2f dB, noise=%s",
        snr_db, actual_snr, Path(noise_path).name,
    )
    return mixed, actual_snr


# ---------------------------------------------------------------------------
# Room impulse response (reverberation)
# ---------------------------------------------------------------------------

def apply_room_ir(
    audio: np.ndarray,
    sr: int,
    ir_files: List[str],
    seed: Optional[int] = None,
) -> np.ndarray:
    """Convolve audio with a random room impulse response (IR).

    Simulates the acoustic effect of recording in a room. A random IR file
    is selected, loaded, resampled to sr, and convolved with the audio using
    FFT-based convolution. The output is truncated to the original length and
    normalised to preserve the original RMS amplitude.

    Reference: Testkonzept Phase 1 — Gruppe B condition B_room_ir.
               scipy.signal.fftconvolve(audio, ir)[:len(audio)].

    Args:
        audio:    1-D float32 audio array.
        sr:       Sample rate in Hz.
        ir_files: List of absolute paths to impulse response .wav files.
        seed:     Random seed for IR file selection.

    Returns:
        Convolved and RMS-normalised 1-D float32 array (same length as input).

    Raises:
        ValueError: If ir_files is empty.
    """
    if not ir_files:
        raise ValueError("ir_files list is empty — cannot apply room IR.")

    rng = random.Random(seed)
    ir_path = rng.choice(ir_files)

    ir, _ = librosa.load(str(ir_path), sr=sr, mono=True, dtype=np.float32)

    if len(ir) == 0:
        logger.warning("Empty IR file: %s. Returning unmodified audio.", ir_path)
        return audio.copy()

    # FFT convolution, truncate to original length
    convolved = fftconvolve(audio, ir)[:len(audio)]

    # Normalise to original RMS to prevent level change
    rms_in = float(np.sqrt(np.mean(audio ** 2)))
    rms_out = float(np.sqrt(np.mean(convolved ** 2)))

    if rms_out > 1e-10 and rms_in > 1e-10:
        convolved = convolved * (rms_in / rms_out)

    convolved = np.clip(convolved, -1.0, 1.0).astype(np.float32)

    logger.debug("apply_room_ir: ir=%s", Path(ir_path).name)
    return convolved


# ---------------------------------------------------------------------------
# MP3 compression (in-memory, no disk write)
# ---------------------------------------------------------------------------

def apply_mp3_compression(
    audio: np.ndarray,
    sr: int,
    bitrate_kbps: int,
) -> np.ndarray:
    """Encode audio as MP3 and decode back, entirely in memory.

    Simulates the artefacts introduced by lossy MP3 compression at a given
    bitrate. Uses pydub for encode/decode via an in-memory BytesIO buffer —
    no temporary files are written to disk.

    The pipeline:
        float32 array → int16 PCM → pydub AudioSegment → MP3 encode (BytesIO)
        → MP3 decode → int16 PCM → float32 array

    Reference: Testkonzept Phase 1 — Gruppe B conditions B_mp3_128, B_mp3_64.

    Args:
        audio:        1-D float32 audio array in [-1, 1].
        sr:           Sample rate in Hz.
        bitrate_kbps: MP3 bitrate in kbps. Typically 128 or 64.

    Returns:
        Re-decoded 1-D float32 array (length may differ slightly from input
        due to MP3 frame-size boundary effects; trimmed/padded to match).
    """
    from pydub import AudioSegment

    # float32 → int16 PCM
    pcm_int16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    pcm_bytes = pcm_int16.tobytes()

    # Build pydub AudioSegment from raw PCM
    segment = AudioSegment(
        data=pcm_bytes,
        sample_width=2,       # 16-bit = 2 bytes
        frame_rate=sr,
        channels=1,
    )

    # Encode to MP3 into a BytesIO buffer
    buf = io.BytesIO()
    segment.export(buf, format="mp3", bitrate=f"{bitrate_kbps}k")
    buf.seek(0)

    # Decode MP3 back to AudioSegment
    decoded_segment = AudioSegment.from_mp3(buf)

    # Extract raw samples and convert back to float32
    pcm_decoded = np.frombuffer(decoded_segment.raw_data, dtype=np.int16)
    audio_out = pcm_decoded.astype(np.float32) / 32768.0

    # Match length to original (MP3 frame padding can add/remove a few samples)
    n_in = len(audio)
    if len(audio_out) >= n_in:
        audio_out = audio_out[:n_in]
    else:
        pad = np.zeros(n_in - len(audio_out), dtype=np.float32)
        audio_out = np.concatenate([audio_out, pad])

    logger.debug("apply_mp3_compression: %d kbps, in=%d, out=%d samples",
                 bitrate_kbps, n_in, len(audio_out))
    return audio_out


# ---------------------------------------------------------------------------
# Combined distortions (Gruppe C)
# ---------------------------------------------------------------------------

def apply_combined(
    audio: np.ndarray,
    sr: int,
    operations: List[Tuple],
) -> np.ndarray:
    """Apply a sequence of distortion operations to audio.

    Each operation is a 2-tuple of (function, kwargs_dict). The function is
    called as function(current_audio, sr, **kwargs). Operations are applied
    in list order, and the output of each step is passed as input to the next.

    Special handling for apply_noise return value: since apply_noise returns
    a (audio, actual_snr) tuple, apply_combined unpacks it automatically.

    Supported operation format:
        (apply_tempo_change,    {"rate": 1.10})
        (apply_noise,           {"snr_db": 5, "noise_files": [...]})
        (apply_room_ir,         {"ir_files": [...]})
        (apply_pitch_shift,     {"semitones": +1})
        (apply_mp3_compression, {"bitrate_kbps": 64})

    Reference: Testkonzept Phase 1 — Gruppe C (C_club, C_radio).

    Args:
        audio:      1-D float32 audio array.
        sr:         Sample rate in Hz.
        operations: Ordered list of (distortion_fn, kwargs) tuples.

    Returns:
        Distorted 1-D float32 array after all operations have been applied.
    """
    current = audio.copy()
    for fn, kwargs in operations:
        result = fn(current, sr, **kwargs)
        # apply_noise returns (audio, snr) — unpack transparently
        if isinstance(result, tuple):
            current = result[0]
        else:
            current = result
    return current.astype(np.float32)


# ---------------------------------------------------------------------------
# WAV export
# ---------------------------------------------------------------------------

def save_wav(audio: np.ndarray, sr: int, path: str) -> None:
    """Write a float32 audio array to a WAV file.

    Creates parent directories automatically if they do not exist.
    Uses soundfile for reliable 32-bit float WAV output.

    Args:
        audio: 1-D float32 audio array.
        sr:    Sample rate in Hz.
        path:  Destination file path (absolute or relative).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    logger.debug("Saved WAV: %s (%d samples, %d Hz)", path, len(audio), sr)
