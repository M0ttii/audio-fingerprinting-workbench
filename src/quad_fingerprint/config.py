"""
Zentrale Konfigurationsdatei für den Quad-basierten Audio-Fingerprinting-Algorithmus.

Alle algorithmischen Parameter sind hier definiert und mit Bezug auf das Paper
kommentiert:
    Sonnleitner & Widmer (2016): "Robust Quad-Based Audio Fingerprinting"
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, Vol. 24, No. 3.

Kein Modul darf Magic Numbers oder hardcoded algorithmische Parameterwerte verwenden.
"""

# ==============================================================================
# I. AUDIO-VORVERARBEITUNG
# Referenz: Section IV ("Feature Extraction")
# ==============================================================================

# Abtastrate für alle Audiodateien.
# Paper: "all audio files are downmixed to one-channel monaural representations
# sampled at 8 kHz." (Section IV)
SAMPLE_RATE: int = 8_000  # Hz

# STFT-Fenstergröße (Hann-Fenster).
# Paper: "Hann-window of size 1024 samples (128 ms)" (Section IV)
N_FFT: int = 1_024  # Samples; entspricht 128 ms bei 8 kHz

# STFT-Hopgröße.
# Paper: "a hop size of 32 samples (4 ms)" (Section IV, Section VII)
# Reduziert gegenüber Vorgängerarbeit [3] von 128 auf 32 Samples für höhere
# Zeitauflösung der Hashes.
HOP_LENGTH: int = 32  # Samples; entspricht 4 ms bei 8 kHz

# Abgeleitete Größe: STFT-Frames pro Sekunde.
# Wird für Umrechnungen zwischen Sekunden und Frame-Indizes verwendet.
FRAMES_PER_SECOND: float = SAMPLE_RATE / HOP_LENGTH  # = 250.0 Frames/s

# ==============================================================================
# II. PEAK-EXTRAKTION — REFERENZ-AUDIO
# Referenz: Section IV-A ("Constructing Quads"), Section V
# ==============================================================================

# Breite (Zeit-Dimension) des Max-Filters für Referenz-Peaks.
# Paper: "extraction of reference peaks is performed with a max-filter width of
# 151 STFT-frames" (Section V, S. 414)
REF_MAX_FILTER_WIDTH: int = 151  # STFT-Frames

# Höhe (Frequenz-Dimension) des Max-Filters für Referenz-Peaks.
# Paper: "a filter height of 75 frequency bins" (Section V, S. 414)
REF_MAX_FILTER_HEIGHT: int = 75  # Frequenz-Bins

# Größe des Min-Filters (quadratisch) zur Eliminierung uniformer Regionen
# (Stille, Klicks, digital erzeugte Töne mit identischen Magnitudewerten).
# Paper: "a min filter of size 3×3" (Section IV-A)
MIN_FILTER_WIDTH: int = 3   # STFT-Frames
MIN_FILTER_HEIGHT: int = 3  # Frequenz-Bins

# ==============================================================================
# III. TOLERANZ-PARAMETER (EPSILON)
# Referenz: Section IV-B ("Query Quad Construction"), Section VIII
# ==============================================================================

# Toleranz für Zeit-Skalierungsverzerrungen (Tempo/Speed).
# Definiert den maximalen relativen Unterschied in der Zeitskala zwischen
# Referenz- und Query-Audio, der noch korrekt erkannt werden soll.
# Paper: "For all experiments in this work, the fingerprinter is configured with
# scale transform tolerances ε_p = ε_t = 0.31." (Section VIII)
# Beispiel: ε_t = 0.31 entspricht Robustheit gegenüber ±31% Tempo-Änderung.
EPSILON_T: float = 0.31  # Zeittoleranz (dimensionslos)

# Toleranz für Frequenz-Skalierungsverzerrungen (Pitch/Speed).
# Paper: "ε_p = ε_t = 0.31" (Section VIII)
# Beispiel: ε_p = 0.31 entspricht Robustheit gegenüber ±31% Pitch-Änderung.
EPSILON_P: float = 0.31  # Frequenztoleranz (dimensionslos)

# ==============================================================================
# IV. PEAK-EXTRAKTION — QUERY-AUDIO (abgeleitet von Referenz-Parametern)
# Referenz: Section IV-B, Eq. (2) und (3)
# ==============================================================================

# Breite des Max-Filters für Query-Peaks (Eq. 2):
#   m_w^query = m_w^ref / (1 + ε_t)
# Query-Peaks werden dichter extrahiert, da bei erhöhtem Tempo die relevanten
# Peaks näher zusammenliegen als im Referenz-Audio.
QUERY_MAX_FILTER_WIDTH: int = round(REF_MAX_FILTER_WIDTH / (1 + EPSILON_T))
# ≈ round(151 / 1.31) = round(115.3) = 115 Frames

# Höhe des Max-Filters für Query-Peaks (Eq. 3):
#   m_h^query = m_h^ref · (1 − ε_p)
# Kleinere Filterhöhe → höhere Frequenz-Auflösung der extrahierten Peaks.
QUERY_MAX_FILTER_HEIGHT: int = 52
# ≈ round(75 · 0.69) = round(51.75) = 52 Bins
# Hinweis: Paper Section VIII nennt explizit 51 Bins — kleinere Differenz durch
# Rundung; bei Bedarf auf 51 setzen.

# ==============================================================================
# V. QUAD-GRUPPIERUNG — REFERENZ-AUDIO
# Referenz: Section IV-A, Section V (S. 414)
# ==============================================================================

# Abstand des Mittelpunkts des Gruppierungsfensters vom Root-Point A (in Sekunden).
# Paper: "center of the grouping window c^ref to be 1.3 seconds from each root
# point A" (Section V, S. 414)
REF_CENTER_C_SEC: float = 1.3  # Sekunden

# Breite des Gruppierungsfensters in Sekunden.
# Paper: "The width r^ref is 0.8 seconds." (Section V, S. 414)
REF_WIDTH_R_SEC: float = 0.8  # Sekunden

# Abgeleitete Größen in STFT-Frames:
REF_CENTER_C_FRAMES: int = round(REF_CENTER_C_SEC * FRAMES_PER_SECOND)   # = 325 Frames
REF_WIDTH_R_FRAMES: int = round(REF_WIDTH_R_SEC * FRAMES_PER_SECOND)     # = 200 Frames

# Fenstergrenzen des Referenz-Gruppierungsfensters (in Frames):
#   r_start^ref = c^ref - r^ref / 2
#   r_end^ref   = c^ref + r^ref / 2
REF_R_START_FRAMES: int = REF_CENTER_C_FRAMES - REF_WIDTH_R_FRAMES // 2  # = 225 Frames
REF_R_END_FRAMES: int = REF_CENTER_C_FRAMES + REF_WIDTH_R_FRAMES // 2    # = 425 Frames

# Maximale Anzahl von Referenz-Quads pro Sekunde Audio.
# Paper: "maximum of q = 9 quads per second of reference audio" (Section V, S. 414)
REF_QUADS_PER_SECOND: int = 9

# ==============================================================================
# VI. QUAD-GRUPPIERUNG — QUERY-AUDIO (abgeleitet via Eq. 4a–4c)
# Referenz: Section IV-B, Eq. (4a)–(4c)
# ==============================================================================

# Fenstergrenzen des Query-Gruppierungsfensters (in Frames):
#   r_start^query = r_start^ref / (1 + ε_t)   [Eq. 4a]
#   r_end^query   = r_end^ref   / (1 − ε_t)   [Eq. 4b]
#   c^query       = (r_start^query + r_end^query) / 2  [Eq. 4c]
QUERY_R_START_FRAMES: float = REF_R_START_FRAMES / (1 + EPSILON_T)  # ≈ 171.8 Frames
QUERY_R_END_FRAMES: float = REF_R_END_FRAMES / (1 - EPSILON_T)       # ≈ 615.9 Frames
QUERY_CENTER_C_FRAMES: float = (QUERY_R_START_FRAMES + QUERY_R_END_FRAMES) / 2  # ≈ 393.8 F

# Maximale Anzahl von Query-Quads pro Sekunde Audio.
# Paper: "extract a number of q ≈ 1500 quads per second of query audio" (Section VIII)
# Höherer Wert als bei Referenz, um bei Tempo-/Pitch-Variationen genügend
# übereinstimmende Quads zu finden.
QUERY_QUADS_PER_SECOND: int = 1_500

# ==============================================================================
# VII. RELEVANTER TEILRAUM (SUBSPACE-CONSTRAINT)
# Referenz: Section IV-C, Eq. (5)
# ==============================================================================

# Untere Grenze der C'x- und D'x-Hash-Komponenten für Referenz-Quads (Eq. 5):
#   C'x_min^ref = D'x_min^ref = (c^ref − r^ref/2) / (c^ref + r^ref/2)
#                             = r_start^ref / r_end^ref
# Query-Quads mit C'x^query < HASH_CX_MIN − SEARCH_RADIUS können verworfen
# werden, da kein Referenz-Quad im Hash-Raum existiert.
# Paper: "44.7% of possible quads could be rejected" (Section IV-C)
HASH_CX_MIN: float = REF_R_START_FRAMES / REF_R_END_FRAMES  # ≈ 0.529

# ==============================================================================
# VIII. DATENBANK-PARAMETER
# Referenz: Section V
# ==============================================================================

# Datentyp für Peak-Koordinaten (float32, 8 Bytes/Peak).
# Paper: "peaks are represented as single precision floating point values"
# (Section IV-A); "each peak is represented by two single precision floats,
# and consumes 8 bytes" (Section V)
PEAK_DTYPE: str = "float32"

# Dateiname der Datenbank-Dateien (relativ zu einem konfigurierbaren DB-Verzeichnis)
PEAKFILE_NAME: str = "peakfile.bin"
REFRECORDS_NAME: str = "refrecords.bin"
FIDINDEX_NAME: str = "fidindex.pkl"
SEARCHTREE_NAME: str = "searchtree.pkl"

# ==============================================================================
# IX. MATCHING — KANDIDATENAUSWAHL
# Referenz: Section VI-A
# ==============================================================================

# Suchradius für Fixed-Radius-Nearest-Neighbor-Suche im 4D-Hash-Raum.
# Paper: "near-neighbour search radius ε_L = 0.01" (Section VIII, Fig. 5)
# Gilt komponentenweise: |C'x^query − C'x^ref| ≤ ε_L usw.
SEARCH_RADIUS: float = 0.01  # ε_L (dimensionslos, im Einheitsquadrat [0,1]^4)

# Schwellenwert für den Fein-Pitch-Kohärenzfilter (Eq. 9):
#   |A_y^query − A_y^ref · s_freq| ≤ ε_pfine
# Paper: "ε_pfine = 1.8, which we determined empirically" (Section VI-A)
FINE_PITCH_COHERENCE_THRESHOLD: float = 1.8  # Frequenz-Bins (ε_pfine)

# ==============================================================================
# X. MATCHING — SEQUENZSCHÄTZUNG
# Referenz: Section VI-B
# ==============================================================================

# Mindestanzahl übereinstimmender Kandidaten pro Datei-ID, damit eine Sequenz
# zur Verifikation weitergeleitet wird.
# Paper: "if match sequences are found for a given file-ID, and their number of
# matched candidates is larger than a threshold value t_s" (Section VI-B);
# "Sequences must contain at least t_s = 4 matches" (Section VIII)
SEQUENCE_MIN_MATCHES: int = 4  # t_s

# Histogramm-Bin-Breite für die Shazam-analoge Sequenzschätzung (in Sekunden).
# Nicht explizit im Paper genannt; empirisch gewählter Wert kompatibel mit
# dem adaptierten Shazam-Histogram-Ansatz (Section VI-B, vgl. Wang 2003 [5]).
HISTOGRAM_BIN_SIZE_SEC: float = 0.5  # Sekunden

# Faktor für varianzbasiertes Outlier-Removal in der Sequenzschätzung.
# Paper: "variance of scale transforms larger than a threshold value" (Section VI-B).
# Kandidaten, deren Skalierungsfaktor mehr als OUTLIER_SIGMA_FACTOR
# Standardabweichungen vom Median abweicht, werden verworfen.
OUTLIER_SIGMA_FACTOR: float = 3.0  # empirisch

# ==============================================================================
# XI. MATCHING — VERIFIKATION
# Referenz: Section VI-C, Eq. (10)–(11)
# ==============================================================================

# Mindest-Verifikationsscore (Anteil korrekt ausgerichteter Referenz-Peaks),
# ab dem eine Sequenz als Match akzeptiert wird.
# Paper: "threshold for the average verification score of sequences is set to
# 0.53" (Section VIII); "we try not to find all, but just a percentage of
# t_min of the nearby reference peaks" (Section VI-C)
VERIFICATION_THRESHOLD: float = 0.53  # t_min (Anteil, dimensionslos)

# Zeitbereich um den Root-Point A des Match-Kandidaten, innerhalb dessen
# Referenz-Peaks für die Verifikation gesucht werden.
# Paper: "peaks in the range of ±1.8s near the reference candidate rootpoint"
# (Section VIII)
VERIFICATION_TIME_RANGE_SEC: float = 1.8  # Sekunden (±)

# Größe des rechteckigen Toleranz-Fensters für die Peak-Ausrichtung in der
# Frequenz-Dimension.
# Paper: "rectangular alignment regions of height 12 frequency bins" (Section VIII,
# Section VI-C: "span 12 frequency bins and 18 STFT frames (0.072s)")
VERIFICATION_FREQ_TOLERANCE: int = 12  # Frequenz-Bins (halbseitig: ±6)

# Größe des rechteckigen Toleranz-Fensters für die Peak-Ausrichtung in der
# Zeit-Dimension.
# Paper: "18 STFT frames (0.072s)" (Section VIII, Section VI-C)
VERIFICATION_TIME_TOLERANCE: int = 18  # STFT-Frames (halbseitig: ±9)

# Mindestanteil der Query-Snippet-Länge, den eine verifizierte Match-Sequenz
# abdecken muss, damit ein Match gemeldet wird.
# Paper: "if the sequence of verified matches for the given file-ID covers at
# least 15% of the query snippet length, we report the match." (Section VI-C)
MATCH_COVERAGE_THRESHOLD: float = 0.15  # (Anteil, dimensionslos)

# ==============================================================================
# XII. EVALUATION
# Referenz: Section VIII
# ==============================================================================

# Standard-Query-Snippet-Länge in Sekunden (für Evaluation).
# Paper: "extract a starting position ... and cut out 20 seconds from the audio"
# (Section VIII)
DEFAULT_QUERY_DURATION_SEC: float = 20.0  # Sekunden

# Skalierungsfaktoren für Robustheitstests (Speed, Tempo, Pitch).
# Paper: "scale distortions in the range from 70% to 130% in steps of
# 5 percentage points" (Section VIII-A)
EVAL_SCALE_FACTORS: list[float] = [
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    1.00,
    1.05, 1.10, 1.15, 1.20, 1.25, 1.30,
]

# SNR-Bereich für Rausch-Robustheitstests (in dB).
# Paper: "SNR ranges from −10 dB to +50 dB in steps of 5 dB" (Section VIII-C)
EVAL_SNR_MIN_DB: float = -10.0
EVAL_SNR_MAX_DB: float = 50.0
EVAL_SNR_STEP_DB: float = 5.0

# ==============================================================================
# XIII. PFADE
# ==============================================================================

import pathlib

# Projekt-Wurzelverzeichnis (zwei Ebenen über dieser Datei)
PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).parent.parent

DATA_DIR: pathlib.Path = PROJECT_ROOT / "data"
REFERENCE_DIR: pathlib.Path = DATA_DIR / "reference"
QUERY_DIR: pathlib.Path = DATA_DIR / "queries"
DB_DIR: pathlib.Path = DATA_DIR / "db"
RESULTS_DIR: pathlib.Path = PROJECT_ROOT / "results"
