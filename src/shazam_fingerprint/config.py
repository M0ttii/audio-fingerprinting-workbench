"""
Zentrale Konfigurationsdatei für den Shazam Audio-Fingerprinting-Algorithmus.

Alle algorithmischen Parameter sind hier definiert. Kein Modul darf Magic Numbers
oder hardcoded Werte verwenden. Jeder Parameter enthält einen Kommentar mit Bezug
zum Whitepaper: Wang, A. (2003). "An Industrial-Strength Audio Search Algorithm."
Proceedings of the 4th International Conference on Music Information Retrieval (ISMIR).
"""

from pathlib import Path

# ==============================================================================
# === Pfade ====================================================================
# ==============================================================================

# Wurzelverzeichnis des Projekts (relativ zu dieser Datei)
PROJECT_ROOT: Path = Path(__file__).parent.parent

# Verzeichnis für Original-Audiodateien (Referenz-Datenbank)
REFERENCE_DIR: Path = PROJECT_ROOT / "data" / "reference"

# Verzeichnis für Query-Audiodateien (ggf. verzerrte Varianten)
QUERY_DIR: Path = PROJECT_ROOT / "data" / "queries"

# Pfad zur serialisierten Fingerprint-Datenbank
DB_PATH: Path = PROJECT_ROOT / "data" / "fingerprint_db.pkl"

# Verzeichnis für Evaluationsergebnisse und Plots
RESULTS_DIR: Path = PROJECT_ROOT / "results"


# ==============================================================================
# === Audio-Vorverarbeitung ====================================================
# ==============================================================================

# Ziel-Abtastrate in Hz.
# Wang (2003) testete ursprünglich mit 8 kHz für GSM-Telefonie. Für die
# Evaluation mit unkomprimiertem Audio (GTZAN, FMA) ist 22050 Hz ein
# guter Kompromiss: volle Frequenzabdeckung bis 11025 Hz (Nyquist), aber
# halb so viele Samples wie CD-Qualität (44100 Hz) → geringerer Rechenaufwand.
# Wang Section 3.1: "The system was tested ... at 8000 samples per second."
SAMPLE_RATE: int = 22050

# Immer Mono konvertieren. Fingerprinting benötigt nur einen Kanal.
# Mehrkanal-Audio enthält redundante Information; durch Mono-Konvertierung
# werden Phase-Artefakte zwischen Kanälen vermieden.
MONO: bool = True

# Ladeumfang: None = gesamte Datei laden.
# Für Query-Generierung kann auf 5/10/15 Sekunden gesetzt werden.
# Wang Section 3.1: "The query was restricted to 10 seconds."
DURATION: float | None = None

# Länge eines Query-Ausschnitts in Sekunden (für die Robustheitsevaluation).
# Wang Section 3.1 zeigt, dass 10 s Queries hohe Erkennungsraten liefern.
QUERY_DURATION_SEC: float = 10.0


# ==============================================================================
# === Spektrogramm (STFT) ======================================================
# ==============================================================================

# FFT-Fenstergröße in Samples.
# Mit SAMPLE_RATE=22050 und N_FFT=4096 ergibt sich eine Frequenzauflösung von
# 22050/4096 ≈ 5.4 Hz pro Bin. Das entspricht der in Wang Section 2.2 erwähnten
# "high-resolution frequency axis" für präzise Peak-Lokalisation.
# Wang: "The spectrogram ... was computed using a 1024-bin frequency axis"
# → deutet auf n_fft=2048 hin. Wir verwenden 4096 für bessere Frequenzauflösung
# bei 22050 Hz Samplerate, da wir keine Bandbreitenbeschränkung durch GSM haben.
N_FFT: int = 4096

# Anzahl Samples zwischen aufeinanderfolgenden STFT-Frames (Hop Size).
# HOP_LENGTH = N_FFT // 2 entspricht 50% Overlap, was ein guter Standard-
# Kompromiss zwischen Zeitauflösung und Rechenaufwand ist.
# Bei SAMPLE_RATE=22050 und HOP_LENGTH=2048: ~10.8 Frames pro Sekunde.
HOP_LENGTH: int = 2048

# Fensterfunktion für die STFT.
# Das Hann-Fenster reduziert spektrales Leck (spectral leakage) und ist
# der Standard für Audio-Analyse. Wang erwähnt keine spezifische Fensterfunktion,
# aber Hann ist implizierter Standard in der STFT-Literatur.
WINDOW: str = "hann"

# Maximale Frequenz in Hz, die für Fingerprinting verwendet wird.
# Frequenzen über diesem Wert werden ignoriert. Musik enthält die meisten
# relevanten Strukturmerkmale unterhalb von 5000 Hz. Wang verwendete bei
# 8 kHz Samplerate (Nyquist: 4000 Hz) ähnliche effektive Bandbreiten.
MAX_FREQUENCY_HZ: float = 5000.0

# Minimale Frequenz in Hz. DC-Anteil und sehr tiefe Frequenzen (< 300 Hz)
# sind weniger stabil gegenüber Verzerrungen und werden ausgeschlossen.
MIN_FREQUENCY_HZ: float = 300.0


# ==============================================================================
# === Peak-Extraktion (Constellation Map) =====================================
# ==============================================================================

# Größe der Nachbarschaft in Frequenz-Bins für maximum_filter().
# Ein Peak bei (f, t) wird akzeptiert, wenn er das Maximum in einem Rechteck
# der Größe (2*FREQ + 1) × (2*TIME + 1) ist.
# Wang Section 2.1: "A time-frequency point is a candidate peak if it has a
# higher energy content than all its neighbors in a region centered around
# the point."
PEAK_NEIGHBORHOOD_SIZE_FREQ: int = 10  # Bins (entspricht ~108 Hz bei 4096er FFT)

# Größe der Nachbarschaft in Zeitframes für maximum_filter().
# Wang Section 2.1: Keine explizite Zahl, aber die Nachbarschaftsgröße
# bestimmt die Dichte der Constellation Map. Größere Werte → weniger Peaks.
PEAK_NEIGHBORHOOD_SIZE_TIME: int = 5  # Frames (entspricht ~1.9 s bei 22050/2048)

# Maximale Anzahl Peaks pro Sekunde (Dichte-Kriterium).
# Zu viele Peaks erhöhen den Hash-Aufwand; zu wenige verringern die Robustheit.
# Wang Section 2.1: "Candidate peaks are chosen according to a density criterion
# in order to assure that the time-frequency strip for the audio file has
# reasonably uniform coverage."
MAX_PEAKS_PER_SECOND: int = 30

# Minimale Amplitude eines Peaks in dB (relativ zum Maximum des Spektrogramms).
# Peaks unterhalb dieses Schwellwerts sind wahrscheinlich Rauschen und werden
# verworfen. Wang Section 2.1: "the highest amplitude peaks are most likely to
# survive the distortions to which the audio has been subjected."
AMPLITUDE_THRESHOLD_DB: float = -60.0


# ==============================================================================
# === Combinatorial Hashing (Fingerprint-Generierung) =========================
# ==============================================================================

# Anzahl Target-Peaks pro Anchor-Peak (Fan-Out).
# Jeder Anchor bildet Paare mit den FAN_OUT nächsten Peaks in seiner Target Zone.
# Höherer Fan-Out → mehr Hashes → robuster, aber mehr Speicher.
# Wang Section 2.2: "fan-out of size F" — Wang demonstriert F=10 als Beispiel.
FAN_OUT: int = 10

# Minimaler Zeitabstand Anchor → Target in Frames.
# Verhindert, dass Anchor und Target zu nah beieinander liegen (unscharfe Paare).
# Wang Section 2.2 zeigt die Target Zone als rechteckigen Bereich rechts vom Anchor.
TARGET_ZONE_T_MIN: int = 1    # Frames (~0.09 s)

# Maximaler Zeitabstand Anchor → Target in Frames.
# Definiert die zeitliche Reichweite der Target Zone. Zu groß → Hashes werden
# unspezifischer; Wang nutzt eine relativ enge Zone für Robustheit.
TARGET_ZONE_T_MAX: int = 50   # Frames (~4.6 s)

# Minimaler Frequenz-Offset (in Bins) vom Anchor zum Target.
# Negativer Wert erlaubt Targets unterhalb des Anchors (bidirektional).
TARGET_ZONE_F_MIN: int = -100  # Bins (~540 Hz bei 4096er FFT)

# Maximaler Frequenz-Offset (in Bins) vom Anchor zum Target.
TARGET_ZONE_F_MAX: int = 100   # Bins (~540 Hz bei 4096er FFT)

# Bits für die Frequenzkomponente im Hash (Anchor-Frequenz und Target-Frequenz).
# Mit 10 Bit können 2^10 = 1024 verschiedene Frequenzwerte kodiert werden.
# Wang Section 2.2: "each hash can be packed into a 32-bit unsigned integer"
# → 10 Bit Anchor-Freq + 10 Bit Target-Freq + 12 Bit Delta-T = 32 Bit.
FREQ_BITS: int = 10

# Bits für die Zeitdifferenz (delta_t) im Hash.
# Mit 12 Bit können Zeitdifferenzen von 0 bis 4095 Frames kodiert werden.
# Bei 22050 Hz / 2048 Hop ≈ 10.8 fps: 4095 Frames ≈ 379 Sekunden maximale
# kodierbare Zeitdifferenz — ausreichend für alle praxisrelevanten Fälle.
DELTA_T_BITS: int = 12

# Maximale Anzahl Frequenz-Bins im Spektrogramm (für Quantisierung).
# N_FFT // 2 + 1 = 2049 Bins für N_FFT=4096. Wird zur Frequenz-Quantisierung
# auf FREQ_BITS verwendet: quantized = int(f_bin * (2**FREQ_BITS - 1) / MAX_FREQ_BIN)
MAX_FREQ_BIN: int = N_FFT // 2  # 2048 (exkl. Nyquist-Bin für Konsistenz)


# ==============================================================================
# === Matching und Scoring =====================================================
# ==============================================================================

# Minimale Anzahl übereinstimmender Hashes, bevor ein Track als Kandidat gilt.
# Filtert Tracks heraus, die nur durch Zufall wenige Hash-Kollisionen aufweisen.
# Verhindert unnötigen Rechenaufwand für das Histogram-Scoring.
MIN_HASH_MATCHES: int = 5

# Minimaler Histogram-Peak-Score für eine positive Erkennung.
# Wang Section 2.3.1: "an acceptable false positive rate is chosen, then a
# threshold score is chosen that meets or exceeds the false-positive criterion."
# Muss experimentell kalibriert werden (Abhängigkeit von Query-Länge und DB-Größe).
MATCH_THRESHOLD: int = 19

# Breite der Histogram-Bins für δt-Werte in Frames.
# Wang Section 2.3: "calculate a histogram of δt values and scan for a peak."
# Bin-Breite = 1 Frame bedeutet exakte Zeitkohärenz wird gefordert (präzise).
# Größere Bin-Breiten tolerieren leichte Tempo-Schwankungen (robuster).
HISTOGRAM_BIN_WIDTH: int = 1


# ==============================================================================
# === Reproduzierbarkeit =======================================================
# ==============================================================================

# Seed für numpy.random — sichert Reproduzierbarkeit bei stochastischen
# Operationen (z.B. falls zufälliges Subsampling bei Peak-Dichte verwendet wird).
RANDOM_SEED: int = 42


# ==============================================================================
# === Logging ==================================================================
# ==============================================================================

# Log-Level für das Projekt (DEBUG, INFO, WARNING, ERROR, CRITICAL).
# Im Produktionsbetrieb: INFO; beim Debuggen: DEBUG.
LOG_LEVEL: str = "DEBUG"

# Log-Format: Zeitstempel, Modul-Name, Level, Nachricht.
LOG_FORMAT: str = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
