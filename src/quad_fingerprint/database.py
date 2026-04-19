"""
Referenz-Datenbank mit vier Datenstrukturen nach Sonnleitner & Widmer, Section V.

Paper: "four data structures that together constitute what we call the reference
database: peakfile, refrecords, fidindex and searchtree." (Section V)

Peakfile:    float32-Koordinaten aller Peaks (8 Bytes/Peak).
Refrecords:  Alle Quad-Records: A, S, Hash, file_id (36 Bytes/Record).
Fidindex:    Mapping file_id ↔ Metadaten + Peakfile-Index.
Searchtree:  scipy.spatial.cKDTree über alle 4D-Hash-Vektoren.

WICHTIG: Searchtree wird EINMALIG nach dem Ingest ALLER Songs gebaut,
nicht nach jedem einzelnen. Aufruf: database.finalize()
"""

import logging
import pathlib
import pickle
import time

import numpy as np
from scipy.spatial import cKDTree

from quad_fingerprint import config
from quad_fingerprint.quad_builder import QuadRecord

logger = logging.getLogger(__name__)

# Structured dtype für Refrecords (36 Bytes pro Record)
# Paper Section V: "A quad record consist of spectral peak A and S,
# the quad hash C', D', and the file-ID ... a size of 36 bytes."
REFRECORD_DTYPE = np.dtype([
    ('a_time', np.float32),       # 4 Bytes — Root-Point A: time
    ('a_freq', np.float32),       # 4 Bytes — Root-Point A: freq
    ('s_time', np.float32),       # 4 Bytes — Quad-Größe S_x = B_x − A_x
    ('s_freq', np.float32),       # 4 Bytes — Quad-Größe S_y = B_y − A_y
    ('hash', np.float32, (4,)),   # 16 Bytes — Hash (C'_x, C'_y, D'_x, D'_y)
    ('file_id', np.uint32),       # 4 Bytes — Datei-ID
])  # Gesamt: 36 Bytes ✓

# Dateinamen für Persistenz
_PEAKFILE = "peakfile.npy"
_REFRECORDS = "refrecords.npy"
_FIDINDEX = "fidindex.pkl"
_SEARCHTREE = "searchtree.pkl"


class ReferenceDatabase:
    """Referenz-Datenbank für den Quad-Fingerprinting-Algorithmus.

    Workflow:
        1. add_file() für jeden Song (Reihenfolge beliebig).
        2. finalize() → baut peakfile, refrecords-Array und Searchtree.
           EINMALIG nach dem Ingest ALLER Songs aufrufen!
        3. save()/load() zum Persistieren/Laden.
        4. query_radius() für Fixed-Radius NN-Suche.
        5. get_peaks_for_file() für Verifikation.

    Paper Section V: Searchtree ist ein räumlicher Index für Fixed-Radius
    Near-Neighbor-Suche. Wir verwenden scipy.spatial.cKDTree als Ersatz
    für die im Paper beschriebene BVH (Bounding Volume Hierarchy).
    """

    def __init__(self) -> None:
        # --- Akkumulationsphase (vor finalize) ---
        self._peak_lists: list[np.ndarray] = []
        self._record_lists: list[np.ndarray] = []
        self._peak_offset: int = 0

        # --- Fidindex ---
        self._fidindex: dict[int, dict] = {}
        self._name_to_fid: dict[str, int] = {}
        self._next_fid: int = 0

        # --- Finalisierte Strukturen ---
        self._peakfile: np.ndarray | None = None       # (N, 2) float32
        self._refrecords: np.ndarray | None = None      # structured array
        self._searchtree: cKDTree | None = None
        self._finalized: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_finalized(self) -> bool:
        """True wenn finalize() aufgerufen wurde."""
        return self._finalized

    @property
    def n_files(self) -> int:
        """Anzahl der indizierten Audiodateien."""
        return len(self._fidindex)

    @property
    def n_records(self) -> int:
        """Gesamtzahl der Quad-Records."""
        if self._refrecords is not None:
            return len(self._refrecords)
        return sum(len(r) for r in self._record_lists)

    @property
    def n_peaks(self) -> int:
        """Gesamtzahl der gespeicherten Peaks."""
        if self._peakfile is not None:
            return len(self._peakfile)
        return sum(len(p) for p in self._peak_lists)

    # ------------------------------------------------------------------
    # Ingest: Dateien hinzufügen
    # ------------------------------------------------------------------

    def add_file(
        self,
        file_name: str,
        peaks: np.ndarray,
        quad_records: list[QuadRecord],
        duration_sec: float = 0.0,
    ) -> int:
        """Fügt Peaks und Quads einer Audiodatei zur Datenbank hinzu.

        Muss VOR finalize() aufgerufen werden.

        Args:
            file_name: Name der Audiodatei (eindeutiger Schlüssel).
            peaks: (N, 2) float32 Peaks [(time, freq), ...].
                Paper: "peaks for each audio file as a contiguous sequence"
            quad_records: QuadRecord-Liste von build_reference_quads().
            duration_sec: Dauer der Audiodatei in Sekunden (für Metadaten).

        Returns:
            Zugewiesene file_id (uint32).

        Raises:
            RuntimeError: Wenn Datenbank bereits finalisiert ist.
            ValueError: Wenn file_name bereits vorhanden.
        """
        if self._finalized:
            raise RuntimeError(
                "Datenbank bereits finalisiert. Keine weiteren Dateien möglich."
            )
        if file_name in self._name_to_fid:
            raise ValueError(f"Datei bereits in Datenbank: {file_name}")

        file_id = self._next_fid
        self._next_fid += 1

        # Peaks speichern (kontiguente Sequenz pro Datei)
        peaks_f32 = np.asarray(peaks, dtype=np.float32)
        if peaks_f32.ndim == 1 and len(peaks_f32) == 0:
            peaks_f32 = np.empty((0, 2), dtype=np.float32)
        n_peaks = len(peaks_f32)
        self._peak_lists.append(peaks_f32)

        # QuadRecords → structured array konvertieren
        n_quads = len(quad_records)
        if n_quads > 0:
            records = np.empty(n_quads, dtype=REFRECORD_DTYPE)
            for i, qr in enumerate(quad_records):
                records[i]['a_time'] = qr.root_point[0]
                records[i]['a_freq'] = qr.root_point[1]
                records[i]['s_time'] = qr.quad_size[0]
                records[i]['s_freq'] = qr.quad_size[1]
                records[i]['hash'] = qr.hash
                records[i]['file_id'] = file_id
            self._record_lists.append(records)

        # Fidindex aktualisieren
        # Paper: "fidindex maps each reference audio file to a unique file-ID
        # and also stores the number of extracted peaks and quads, along with
        # other meta data." (Section V)
        self._fidindex[file_id] = {
            'file_name': file_name,
            'duration_sec': duration_sec,
            'n_peaks': n_peaks,
            'n_quads': n_quads,
            'peak_start_idx': self._peak_offset,
        }
        self._name_to_fid[file_name] = file_id
        self._peak_offset += n_peaks

        logger.debug(
            "Datei hinzugefügt: '%s' → file_id=%d | %d Peaks, %d Quads",
            file_name, file_id, n_peaks, n_quads,
        )
        return file_id

    # ------------------------------------------------------------------
    # Finalisierung: Strukturen aufbauen
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Baut peakfile, refrecords-Array und Searchtree auf.

        MUSS nach allen add_file()-Aufrufen und VOR allen Queries aufgerufen
        werden. Der Searchtree wird hier EINMALIG gebaut.

        Paper Section V: Der Searchtree indexiert alle Hash-Vektoren für
        effiziente Fixed-Radius Near-Neighbor-Suche.
        """
        if self._finalized:
            logger.warning("Datenbank bereits finalisiert.")
            return

        t_start = time.perf_counter()

        # 1. Peakfile: Alle Peaks konkatenieren
        if self._peak_lists:
            self._peakfile = np.concatenate(self._peak_lists, axis=0)
        else:
            self._peakfile = np.empty((0, 2), dtype=np.float32)
        self._peak_lists.clear()

        # 2. Refrecords: Alle Records konkatenieren
        if self._record_lists:
            self._refrecords = np.concatenate(self._record_lists)
        else:
            self._refrecords = np.empty(0, dtype=REFRECORD_DTYPE)
        self._record_lists.clear()

        # 3. Searchtree: cKDTree über alle 4D-Hash-Vektoren
        # Paper: "searchtree is used to perform efficient fixed-radius
        # near neighbour searches of quad hashes." (Section V)
        if len(self._refrecords) > 0:
            hashes = np.ascontiguousarray(
                self._refrecords['hash'], dtype=np.float64
            )
            self._searchtree = cKDTree(hashes)
        else:
            self._searchtree = None

        self._finalized = True
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "Datenbank finalisiert: %d Dateien | %d Peaks (%.2f MB) | "
            "%d Records (%.2f MB) | Searchtree in %.0f ms",
            self.n_files,
            self.n_peaks, self.n_peaks * 8 / (1024 ** 2),
            self.n_records, self.n_records * 36 / (1024 ** 2),
            elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Abfragen
    # ------------------------------------------------------------------

    def query_radius(
        self,
        query_hashes: np.ndarray,
        radius: float = config.SEARCH_RADIUS,
    ) -> list[np.ndarray]:
        """Fixed-Radius Near-Neighbor-Suche im 4D Hash-Raum.

        Verwendet L∞ (Chebyshev) Distanz → pro-Komponente:
            |C'_x^query − C'_x^ref| ≤ ε_L  etc.
        Das entspricht der Axis-Aligned Bounding-Box-Suche im Paper.

        Paper Section VI-A: "For each query quad hash a fixed-radius near
        neighbour search in the searchtree is performed."

        Args:
            query_hashes: (M, 4) float32/64 Query-Hash-Vektoren.
            radius: Suchradius ε_L (Standard: config.SEARCH_RADIUS = 0.01).

        Returns:
            Liste von M np.ndarray, jeweils mit Refrecord-Indizes der
            Rohkandidaten. Leere Arrays wenn keine Treffer.

        Raises:
            RuntimeError: Wenn Datenbank nicht finalisiert.
        """
        self._ensure_finalized()

        if self._searchtree is None or len(query_hashes) == 0:
            return [np.empty(0, dtype=np.intp) for _ in range(len(query_hashes))]

        # cKDTree.query_ball_point mit L∞-Norm (Chebyshev)
        results = self._searchtree.query_ball_point(
            np.asarray(query_hashes, dtype=np.float64),
            r=radius,
            p=np.inf,
        )

        # Ergebnis normalisieren: konsistent np.ndarray pro Query
        if isinstance(results, np.ndarray) and results.dtype == object:
            return [np.array(r, dtype=np.intp) for r in results]
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            return [np.array(r, dtype=np.intp) for r in results]
        # Einzelner Query-Fall
        if isinstance(results, list) and (
            len(results) == 0 or isinstance(results[0], (int, np.integer))
        ):
            return [np.array(results, dtype=np.intp)]
        return [np.array(r, dtype=np.intp) for r in results]

    def get_refrecords_by_indices(self, indices: np.ndarray) -> np.ndarray:
        """Gibt Refrecords für gegebene Indizes zurück.

        Args:
            indices: Array von Refrecord-Indizes.

        Returns:
            Structured array (Slice) der angeforderten Records.
        """
        self._ensure_finalized()
        return self._refrecords[indices]

    def get_peaks_for_file(self, file_id: int) -> np.ndarray:
        """Gibt alle Peaks einer Datei zurück (für Verifikation).

        Paper Section VI-C: "retrieve [peaks] via a lookup for the peaks
        of the file-ID in the peakfile ... and then bisecting the time values."

        Args:
            file_id: Datei-ID.

        Returns:
            (N, 2) float32 Peaks [(time, freq), ...], nach time sortiert.

        Raises:
            KeyError: Wenn file_id unbekannt.
        """
        self._ensure_finalized()
        if file_id not in self._fidindex:
            raise KeyError(f"Unbekannte file_id: {file_id}")
        info = self._fidindex[file_id]
        start = info['peak_start_idx']
        end = start + info['n_peaks']
        return self._peakfile[start:end]

    def get_file_name(self, file_id: int) -> str:
        """Gibt den Dateinamen für eine file_id zurück.

        Raises:
            KeyError: Wenn file_id unbekannt.
        """
        if file_id not in self._fidindex:
            raise KeyError(f"Unbekannte file_id: {file_id}")
        return self._fidindex[file_id]['file_name']

    def get_file_id(self, file_name: str) -> int:
        """Gibt die file_id für einen Dateinamen zurück.

        Raises:
            KeyError: Wenn Datei nicht in Datenbank.
        """
        if file_name not in self._name_to_fid:
            raise KeyError(f"Datei nicht in Datenbank: {file_name}")
        return self._name_to_fid[file_name]

    def get_file_info(self, file_id: int) -> dict:
        """Gibt die Metadaten einer Datei zurück (aus fidindex).

        Returns:
            Dict mit: file_name, duration_sec, n_peaks, n_quads, peak_start_idx.
        """
        if file_id not in self._fidindex:
            raise KeyError(f"Unbekannte file_id: {file_id}")
        return dict(self._fidindex[file_id])

    # ------------------------------------------------------------------
    # Persistenz
    # ------------------------------------------------------------------

    def save(self, db_dir: str | pathlib.Path) -> None:
        """Speichert die gesamte Datenbank auf Disk.

        Args:
            db_dir: Zielverzeichnis (wird angelegt falls nötig).

        Raises:
            RuntimeError: Wenn Datenbank nicht finalisiert.
        """
        self._ensure_finalized()
        db_dir = pathlib.Path(db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)

        np.save(db_dir / _PEAKFILE, self._peakfile)
        np.save(db_dir / _REFRECORDS, self._refrecords)

        with open(db_dir / _FIDINDEX, 'wb') as f:
            pickle.dump({
                'fidindex': self._fidindex,
                'name_to_fid': self._name_to_fid,
                'next_fid': self._next_fid,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(db_dir / _SEARCHTREE, 'wb') as f:
            pickle.dump(self._searchtree, f, protocol=pickle.HIGHEST_PROTOCOL)

        total_bytes = sum(
            (db_dir / name).stat().st_size
            for name in [_PEAKFILE, _REFRECORDS, _FIDINDEX, _SEARCHTREE]
            if (db_dir / name).exists()
        )
        logger.info("Datenbank gespeichert: %s (%.2f MB)", db_dir, total_bytes / 1024**2)

    @classmethod
    def load(cls, db_dir: str | pathlib.Path) -> 'ReferenceDatabase':
        """Lädt eine gespeicherte Datenbank von Disk.

        Args:
            db_dir: Verzeichnis mit den DB-Dateien.

        Returns:
            Geladene und finalisierte ReferenceDatabase.

        Raises:
            FileNotFoundError: Wenn DB-Dateien fehlen.
        """
        db_dir = pathlib.Path(db_dir)

        # np.save fügt automatisch .npy an — beim Laden berücksichtigen
        peakfile_path = db_dir / _PEAKFILE
        if not peakfile_path.exists() and (db_dir / (_PEAKFILE + ".npy")).exists():
            peakfile_path = db_dir / (_PEAKFILE + ".npy")
        refrecords_path = db_dir / _REFRECORDS
        if not refrecords_path.exists() and (db_dir / (_REFRECORDS + ".npy")).exists():
            refrecords_path = db_dir / (_REFRECORDS + ".npy")

        db = cls()
        db._peakfile = np.load(str(peakfile_path), allow_pickle=False)
        db._refrecords = np.load(str(refrecords_path), allow_pickle=False)

        with open(db_dir / _FIDINDEX, 'rb') as f:
            state = pickle.load(f)  # noqa: S301
        db._fidindex = state['fidindex']
        db._name_to_fid = state['name_to_fid']
        db._next_fid = state['next_fid']

        with open(db_dir / _SEARCHTREE, 'rb') as f:
            db._searchtree = pickle.load(f)  # noqa: S301

        db._finalized = True
        db._peak_offset = len(db._peakfile)

        logger.info(
            "Datenbank geladen: %s | %d Dateien, %d Records, %d Peaks",
            db_dir, db.n_files, db.n_records, db.n_peaks,
        )
        return db

    # ------------------------------------------------------------------
    # Speicherstatistiken
    # ------------------------------------------------------------------

    def memory_usage_mb(self) -> dict[str, float]:
        """Gibt den Speicherbedarf der Datenbank-Komponenten zurück (in MB).

        Paper Section V: "reference database has a size of 9.85 GB" für
        100.000 Songs (~1.8% der Audiodaten).

        Returns:
            Dict mit: peakfile_mb, refrecords_mb, searchtree_est_mb, total_mb.
        """
        mb = 1024.0 ** 2
        peak_mb = (self._peakfile.nbytes / mb) if self._peakfile is not None else 0
        rec_mb = (self._refrecords.nbytes / mb) if self._refrecords is not None else 0
        # cKDTree Speicher: ungefähr 2–3× der Daten (empirisch)
        tree_mb = (self.n_records * 4 * 8 * 2.5 / mb) if self._searchtree else 0
        return {
            'peakfile_mb': round(peak_mb, 4),
            'refrecords_mb': round(rec_mb, 4),
            'searchtree_est_mb': round(tree_mb, 4),
            'total_mb': round(peak_mb + rec_mb + tree_mb, 4),
        }

    # ------------------------------------------------------------------
    # Interne Hilfsfunktionen
    # ------------------------------------------------------------------

    def _ensure_finalized(self) -> None:
        """Prüft, ob die Datenbank finalisiert ist."""
        if not self._finalized:
            raise RuntimeError(
                "Datenbank nicht finalisiert. Bitte finalize() aufrufen."
            )

    def __repr__(self) -> str:
        state = "finalisiert" if self._finalized else "nicht finalisiert"
        return (
            f"ReferenceDatabase({state}: "
            f"{self.n_files} Dateien, {self.n_records} Records, "
            f"{self.n_peaks} Peaks)"
        )
