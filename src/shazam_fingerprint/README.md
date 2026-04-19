# Shazam Audio-Fingerprinting

Implementierung des Shazam-Algorithmus nach Wang (2003) als Baseline-System
für eine Bachelorarbeit über Audio-Fingerprinting-Verfahren.

> Wang, A. L. (2003). *An Industrial-Strength Audio Search Algorithm.*
> Proceedings of the 4th International Conference on Music Information
> Retrieval (ISMIR), Baltimore, MD, USA.

---

## Projektbeschreibung

Dieses Modul implementiert den landmark-basierten Audio-Fingerprinting-Algorithmus,
der von Shazam zur Musikerkennung eingesetzt wird. Der Algorithmus extrahiert
charakteristische Zeitfrequenz-Punkte (sog. *Constellation Map*) aus einem
Spektrogramm, kombiniert diese paarweise zu kompakten 32-bit-Hashes und
speichert sie in einer Hash-Tabelle. Eine unbekannte Audio-Aufnahme wird
erkannt, indem ihre Hashes gegen die Datenbank abgeglichen und per
Zeitkohärenz-Scoring das beste Ergebnis bestimmt wird.

Das Projekt dient als Baseline in einer vergleichenden Studie klassischer
und moderner Audio-Fingerprinting-Verfahren. Die Evaluation erfolgt auf
zwei Dimensionen:

- **Robustheit**: Erkennungsrate unter verzerrten Bedingungen (Rauschen,
  Lautstärke, Kompression, Pitch-/Tempo-Verschiebung)
- **Effizienz**: Fingerprint-Generierungszeit, Matching-Zeit, Speicherbedarf

**Forschungsfrage:** Wie unterscheiden sich klassische Audio-Fingerprinting-
Algorithmen – insbesondere Shazams landmark-basierter Ansatz – in ihrer
Robustheit und Effizienz von neueren Verfahren?

---

## Projektstruktur

```
shazam_fingerprint/
├── config.py          # Zentrale Konfiguration aller Parameter
├── audio_loader.py    # Audiodateien laden und normalisieren
├── spectrogram.py     # STFT-Spektrogramm berechnen
├── peak_finder.py     # Constellation Map: lokale Maxima extrahieren
├── fingerprint.py     # Combinatorial Hashing (Anchor + Target Zone)
├── database.py        # Fingerprint-Datenbank (In-Memory Hash-Tabelle)
├── matcher.py         # Hash-Lookup und Zeitkohärenz-Scoring
├── evaluate.py        # Robustheit- und Effizienz-Metriken
├── pipeline.py        # End-to-End-Pipeline (Ingest, Query, Evaluation)
├── visualization.py   # Plots für die Bachelorarbeit
└── tests/
    ├── test_spectrogram.py
    ├── test_peak_finder.py
    ├── test_fingerprint.py
    ├── test_matcher.py
    └── test_pipeline.py
```

---

## Installation

### Voraussetzungen

- Python 3.10 oder neuer
- pip

### Schritte

```bash
# Repository klonen
git clone <repo-url>
cd shazam

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt

# Paket im Entwicklungsmodus installieren
pip install -e .
```

---

## Quickstart

### 1. Referenz-Songs in die Datenbank laden

```python
from shazam_fingerprint.database import FingerprintDatabase
from shazam_fingerprint.pipeline import ingest_directory

db = FingerprintDatabase()
stats = ingest_directory("data/reference/", db)
print(f"{stats['processed']} Songs, {stats['total_hashes']} Hashes")

# Datenbank für spätere Nutzung speichern
db.save()
```

### 2. Unbekanntes Audio erkennen

```python
from shazam_fingerprint.pipeline import query

result = query("data/queries/unknown_clip.wav", db)

if result.match_found:
    print(f"Erkannt: {result.best_match}  (Score: {result.best_score})")
else:
    print("Kein Match gefunden.")
```

### 3. Ausschnitt abfragen (z. B. 10 Sekunden ab Sekunde 30)

```python
result = query("data/queries/unknown_clip.wav", db,
               start_sec=30.0, duration_sec=10.0)
```

### 4. Gespeicherte Datenbank laden

```python
db = FingerprintDatabase()
db.load()   # lädt aus data/fingerprint_db.pkl (config.DB_PATH)
```

### 5. Robustheit evaluieren

```python
from shazam_fingerprint.pipeline import evaluate_robustness

report = evaluate_robustness(
    query_dir="data/queries/",
    database=db,
    duration_sec=10.0,
)
print(f"Erkennungsrate: {report.recognition_rate:.1%}")
print(f"False-Negative-Rate: {report.false_negative_rate:.1%}")
print(f"Avg. Query-Zeit: {report.avg_query_time_ms:.1f} ms")
```

### 6. Tests ausführen

```bash
pytest shazam_fingerprint/tests/ -v
```

---

## Parameter-Erklärung (`config.py`)

Alle algorithmischen Parameter sind zentral in `config.py` definiert.
Kein Modul verwendet hardcodierte Werte.

### Audio-Vorverarbeitung

| Parameter | Standardwert | Bedeutung |
|---|---|---|
| `SAMPLE_RATE` | `22050` | Ziel-Abtastrate in Hz. Wang (2003) verwendete 8 kHz für GSM; 22050 Hz deckt den für Musik relevanten Frequenzbereich bis 11 kHz ab. |
| `MONO` | `True` | Immer Mono konvertieren. Fingerprinting benötigt nur einen Kanal. |
| `DURATION` | `None` | Gesamte Datei laden. Für Queries ggf. auf 10 s setzen (vgl. Wang Section 3.1). |
| `QUERY_DURATION_SEC` | `10.0` | Standardlänge eines Query-Ausschnitts. Wang Section 3.1: "The query was restricted to 10 seconds." |

### Spektrogramm (STFT)

| Parameter | Standardwert | Bedeutung |
|---|---|---|
| `N_FFT` | `4096` | FFT-Fenstergröße. Ergibt ~5,4 Hz Frequenzauflösung bei 22050 Hz. Wang Section 2.2 deutet auf n_fft=2048 hin; 4096 liefert bessere Auflösung. |
| `HOP_LENGTH` | `2048` | 50 % Overlap. Guter Kompromiss zwischen Zeitauflösung und Rechenaufwand. |
| `WINDOW` | `"hann"` | Hann-Fensterfunktion reduziert spektrales Leck (standard für Audio-STFT). |
| `MIN_FREQUENCY_HZ` | `300.0` | Untere Frequenzgrenze. DC-Anteil und tiefe Frequenzen sind weniger robust gegenüber Verzerrungen. |
| `MAX_FREQUENCY_HZ` | `5000.0` | Obere Frequenzgrenze. Musik enthält die meisten relevanten Strukturmerkmale unterhalb von 5 kHz. |

### Peak-Extraktion (Constellation Map)

| Parameter | Standardwert | Bedeutung |
|---|---|---|
| `PEAK_NEIGHBORHOOD_SIZE_FREQ` | `20` | Nachbarschaftsgröße in Frequenz-Bins für `maximum_filter()`. Wang Section 2.1: Peak muss größer als alle Nachbarn in einem Rechteck sein. |
| `PEAK_NEIGHBORHOOD_SIZE_TIME` | `20` | Nachbarschaftsgröße in Zeitframes für `maximum_filter()`. |
| `MAX_PEAKS_PER_SECOND` | `30` | Dichte-Kriterium: max. Peaks pro 1-Sekunden-Segment. Wang Section 2.1: "reasonably uniform coverage." |
| `AMPLITUDE_THRESHOLD_DB` | `-60.0` | Minimum-Amplitude in dB. Peaks darunter gelten als Rauschen. Wang Section 2.1: "highest amplitude peaks are most likely to survive distortions." |

### Combinatorial Hashing

| Parameter | Standardwert | Bedeutung |
|---|---|---|
| `FAN_OUT` | `10` | Anzahl Target-Peaks pro Anchor. Wang Section 2.2: "fan-out of size F=10." |
| `TARGET_ZONE_T_MIN` | `1` | Minimaler Zeitabstand Anchor→Target in Frames. |
| `TARGET_ZONE_T_MAX` | `50` | Maximaler Zeitabstand Anchor→Target in Frames (~4,6 s). |
| `TARGET_ZONE_F_MIN` | `-100` | Minimaler Frequenz-Offset in Bins (negativ = Target darf tiefer liegen). |
| `TARGET_ZONE_F_MAX` | `100` | Maximaler Frequenz-Offset in Bins. |
| `FREQ_BITS` | `10` | Bits für Frequenzkomponente im Hash (0–1023). |
| `DELTA_T_BITS` | `12` | Bits für Zeitdifferenz im Hash (0–4095). |

Hash-Layout nach Wang Section 2.2:
```
[f_anchor : 10 bit] [f_target : 10 bit] [delta_t : 12 bit]  =  32 bit
```

### Matching und Scoring

| Parameter | Standardwert | Bedeutung |
|---|---|---|
| `MIN_HASH_MATCHES` | `5` | Mindestanzahl Hash-Treffer, bevor ein Song fürs Histogram-Scoring qualifiziert. |
| `MATCH_THRESHOLD` | `10` | Mindestscore für einen positiven Match. Wang Section 2.3.1: muss experimentell kalibriert werden. |
| `HISTOGRAM_BIN_WIDTH` | `1` | Breite der δt-Histogram-Bins in Frames. Wert 1 fordert exakte Zeitkohärenz. |

---

## Theoretische Grundlage

Alle algorithmischen Entscheidungen basieren auf:

> Wang, A. L. (2003). *An Industrial-Strength Audio Search Algorithm.*
> Proceedings of the 4th International Conference on Music Information
> Retrieval (ISMIR). [PDF: `docs/wang2003.pdf`]

Relevante Abschnitte:
- **Section 2.1** — Constellation Map und Peak-Extraktion
- **Section 2.2** — Combinatorial Hashing, Target Zone, Fan-Out, 32-bit-Hash
- **Section 2.3** — Searching and Scoring, δt-Histogram
- **Section 2.3.1** — False-Positive-Rate und Threshold-Kalibrierung
- **Section 3.1** — Experimentelle Evaluation, Query-Länge (10 s)

---

## Evaluation (Bachelorarbeit)

Die Evaluations-Pipeline (`evaluate_robustness`) berechnet:

**Robustheit**
- `recognition_rate` — Anteil korrekt erkannter Queries
- `false_negative_rate` — nicht erkannte Queries (Song in DB vorhanden)
- `false_positive_rate` — Falscherkennungen (Song nicht in DB)

**Effizienz**
- `avg_fingerprint_time_ms` — Durchschnittliche Fingerprint-Zeit pro Query
- `avg_query_time_ms` — Durchschnittliche Matching-Zeit pro Query
- `db_memory_mb` — Speicherbedarf der Datenbank
- `hashes_per_second` — Fingerprint-Durchsatz

Getestete Verzerrungen: Weißes Rauschen, Lautstärke-Variation, MP3-Kompression,
Pitch-Shift, Tempo-Shift.

Audiodatensätze: [GTZAN](http://marsyas.info/downloads/datasets.html),
[FMA](https://github.com/mdeff/fma), [MUSAN](https://www.openslr.org/17/).
