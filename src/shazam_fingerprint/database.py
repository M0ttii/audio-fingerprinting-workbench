"""
Speichereffiziente Fingerprint-Datenbank auf Basis sortierter NumPy-Arrays.

Ersetzt die Python-Dict-Implementierung, die bei 8.000 Songs ~50-80 GB RAM
benötigte (500M Einträge × ~100 Bytes Python-Overhead pro Tuple).

Neue Struktur (nach Finalisierung):
    _hashes_arr:   uint32[N] sortiert   — 4 Bytes/Eintrag
    _song_ids_arr: uint32[N]            — 4 Bytes/Eintrag
    _times_arr:    uint32[N]            — 4 Bytes/Eintrag
    → 12 Bytes/Eintrag total

Bei 500M Einträgen: ~6 GB statt ~50 GB.

Zwei Phasen:
    Build-Phase:  insert() sammelt Daten als numpy-Chunks pro Song.
                  Kein Python-Int-Overhead (im Gegensatz zu list[tuple[int,int]]).
    Query-Phase:  Beim ersten lookup() werden alle Chunks per np.concatenate
                  zusammengefügt, nach Hash-Wert sortiert und danach
                  via np.searchsorted() in O(log N) durchsucht.
                  insert() ist nach der Finalisierung nicht mehr möglich.

Persistenz: np.savez_compressed() statt pickle — deutlich effizienter.

Public API ist identisch zur Dict-basierten Vorgänger-Implementierung.

Wang: "To create a database index, the above operation is carried out on each
track in a database to generate a corresponding list of hashes and their
associated offset times. Track IDs may also be appended to the small data structs."
"""

import logging
from pathlib import Path

import numpy as np

from shazam_fingerprint import config

logger = logging.getLogger(__name__)


class FingerprintDatabase:
    """Speichereffiziente Fingerprint-Datenbank auf Basis sortierter NumPy-Arrays.

    Kernstruktur nach Finalisierung:
        _hashes_arr:   uint32[N] — Hash-Werte, aufsteigend sortiert
        _song_ids_arr: uint32[N] — numerische Song-ID je Eintrag
        _times_arr:    uint32[N] — Anchor-Zeit (Frame-Index) je Eintrag

    Song-ID-Mapping: str ↔ uint32 via _name_to_id / _id_to_name.

    Lookup via np.searchsorted() in O(log N) — Speicher: ~12 Bytes/Eintrag.
    """

    def __init__(self) -> None:
        # Build-Phase: numpy-Chunks pro Song (kein Python-Int-Overhead)
        self._chunks_hashes: list[np.ndarray] = []   # je Song: uint32[M]
        self._chunks_sids:   list[np.ndarray] = []   # je Song: uint32[M]
        self._chunks_times:  list[np.ndarray] = []   # je Song: uint32[M]

        # Song-ID-Mapping: str → uint32 und zurück
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: list[str] = []
        self._song_ids: set[str] = set()  # für __contains__

        # Query-Phase: finalisierte Arrays (None bis _finalize() aufgerufen)
        self._hashes_arr:   np.ndarray | None = None
        self._song_ids_arr: np.ndarray | None = None
        self._times_arr:    np.ndarray | None = None
        self._num_unique_hashes: int = 0  # wird während _finalize() gesetzt
        self._finalized: bool = False

    # ------------------------------------------------------------------
    # Schreib-Operationen
    # ------------------------------------------------------------------

    def insert(self, song_id: str, fingerprints: list[tuple[int, int]]) -> None:
        """Fügt alle Fingerprints eines Songs in die Datenbank ein.

        Speichert die Daten als uint32-NumPy-Arrays (kein Python-Int-Overhead).
        Nach dem ersten lookup()-Aufruf (Finalisierung) ist insert() gesperrt.

        Wang: "Track IDs may also be appended to the small data structs."

        Args:
            song_id: Eindeutiger Bezeichner des Songs (z.B. "000052").
            fingerprints: Liste von (hash_value, anchor_time)-Tupeln aus
                fingerprint.generate_fingerprints().

        Raises:
            ValueError: Wenn song_id leer oder fingerprints leer ist.
            RuntimeError: Nach dem ersten lookup()-Aufruf (Datenbank finalisiert).
        """
        if self._finalized:
            raise RuntimeError(
                "insert() ist nach dem ersten lookup() nicht möglich — "
                "die Datenbank ist finalisiert."
            )
        if not song_id:
            raise ValueError("song_id darf nicht leer sein.")
        if not fingerprints:
            raise ValueError(f"Keine Fingerprints für '{song_id}' übergeben.")

        # Song-ID-Integer zuweisen (falls neuer Song)
        if song_id not in self._name_to_id:
            sid_int = len(self._id_to_name)
            self._name_to_id[song_id] = sid_int
            self._id_to_name.append(song_id)
        sid_int = self._name_to_id[song_id]
        self._song_ids.add(song_id)

        # Fingerprints als uint32-Arrays ablegen
        fp_arr = np.asarray(fingerprints, dtype=np.uint32)  # (M, 2)
        self._chunks_hashes.append(fp_arr[:, 0])
        self._chunks_sids.append(np.full(len(fp_arr), sid_int, dtype=np.uint32))
        self._chunks_times.append(fp_arr[:, 1])

        logger.info(
            "Eingefügt: '%s' (id=%d) | %d Hashes | Songs gesamt: %d",
            song_id, sid_int, len(fp_arr), len(self._song_ids),
        )

    def clear(self) -> None:
        """Leert die Datenbank vollständig und setzt in die Build-Phase zurück."""
        self._chunks_hashes.clear()
        self._chunks_sids.clear()
        self._chunks_times.clear()
        self._name_to_id.clear()
        self._id_to_name.clear()
        self._song_ids.clear()
        self._hashes_arr   = None
        self._song_ids_arr = None
        self._times_arr    = None
        self._num_unique_hashes = 0
        self._finalized = False
        logger.info("Datenbank geleert.")

    # ------------------------------------------------------------------
    # Lese-Operationen
    # ------------------------------------------------------------------

    def lookup(self, hash_value: int) -> list[tuple[str, int]]:
        """Gibt alle (song_id, anchor_time)-Einträge für einen Hash zurück.

        Beim ersten Aufruf: Automatische Finalisierung (Chunks → sortierte Arrays).
        Lookup via np.searchsorted() in O(log N).

        Wang: "The time pairs are distributed into bins according to the
        track ID associated with the matching database hash."

        Args:
            hash_value: 32-bit unsigned integer Hash aus fingerprint._compute_hash().

        Returns:
            Liste von (song_id_str, anchor_time)-Tupeln. Leere Liste wenn kein Eintrag.
        """
        if not self._finalized:
            self._finalize()

        if self._hashes_arr is None or len(self._hashes_arr) == 0:
            return []

        h = np.uint32(hash_value)
        lo = int(np.searchsorted(self._hashes_arr, h, side='left'))
        hi = int(np.searchsorted(self._hashes_arr, h, side='right'))

        if lo >= hi:
            return []

        return [
            (self._id_to_name[int(self._song_ids_arr[i])], int(self._times_arr[i]))
            for i in range(lo, hi)
        ]

    def get_stats(self) -> dict:
        """Berechnet Statistiken über den aktuellen Datenbankzustand.

        Returns:
            Dict mit: num_songs, num_hashes, num_entries, avg_hashes_per_song,
            avg_entries_per_hash, memory_mb.
        """
        num_songs = len(self._song_ids)

        if self._finalized and self._hashes_arr is not None:
            num_entries = len(self._hashes_arr)
            num_hashes  = self._num_unique_hashes
            memory_bytes = (
                self._hashes_arr.nbytes
                + self._song_ids_arr.nbytes
                + self._times_arr.nbytes
            )
        else:
            # Build-Phase: Schätzung aus Chunks
            num_entries  = sum(len(c) for c in self._chunks_hashes)
            num_hashes   = 0  # noch unbekannt
            memory_bytes = sum(
                c.nbytes
                for c in self._chunks_hashes + self._chunks_sids + self._chunks_times
            )

        return {
            "num_songs":           num_songs,
            "num_hashes":          num_hashes,
            "num_entries":         num_entries,
            "avg_hashes_per_song": num_entries / num_songs if num_songs > 0 else 0.0,
            "avg_entries_per_hash": num_entries / num_hashes if num_hashes > 0 else 0.0,
            "memory_mb":           memory_bytes / (1024 ** 2),
        }

    # ------------------------------------------------------------------
    # Persistenz
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Serialisiert die Datenbank mit np.savez_compressed().

        Deutlich effizienter als pickle: komprimierte Arrays statt
        Python-Objektgraph.

        Args:
            path: Zielpfad (wird automatisch mit .npz-Suffix gespeichert).
                  Verwendet config.DB_PATH (mit .npz) wenn None.

        Raises:
            RuntimeError: Wenn die Datenbank leer ist.
        """
        if not self._finalized:
            self._finalize()

        if self._hashes_arr is None:
            raise RuntimeError("Datenbank ist leer — nichts zu speichern.")

        target = _resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            target,
            hashes=self._hashes_arr,
            song_ids=self._song_ids_arr,
            times=self._times_arr,
            id_to_name=np.array(self._id_to_name, dtype=object),
        )

        stats = self.get_stats()
        logger.info(
            "Datenbank gespeichert: '%s' | %d Songs | %d Einträge | %.2f MB",
            target.with_suffix('.npz'),
            stats["num_songs"],
            stats["num_entries"],
            stats["memory_mb"],
        )

    def load(self, path: str | Path | None = None) -> None:
        """Lädt eine zuvor gespeicherte Datenbank aus einer .npz-Datei.

        Überschreibt den aktuellen Inhalt der Datenbank vollständig.

        Args:
            path: Pfad zur .npz-Datei. Verwendet config.DB_PATH (mit .npz) wenn None.

        Raises:
            FileNotFoundError: Wenn die Datei nicht existiert.
            ValueError: Wenn die Datei ein unbekanntes Format hat.
        """
        target = _resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Datenbankdatei nicht gefunden: {target}")

        data = np.load(target, allow_pickle=True)

        required = ('hashes', 'song_ids', 'times', 'id_to_name')
        for key in required:
            if key not in data:
                raise ValueError(f"Fehlendes Feld '{key}' in '{target}'.")

        self._hashes_arr   = data['hashes'].astype(np.uint32)
        self._song_ids_arr = data['song_ids'].astype(np.uint32)
        self._times_arr    = data['times'].astype(np.uint32)
        self._id_to_name   = [str(s) for s in data['id_to_name']]
        self._name_to_id   = {name: i for i, name in enumerate(self._id_to_name)}
        self._song_ids     = set(self._id_to_name)
        self._num_unique_hashes = _count_unique_sorted(self._hashes_arr)
        self._finalized    = True

        # Build-Chunks leeren (irrelevant nach Load)
        self._chunks_hashes.clear()
        self._chunks_sids.clear()
        self._chunks_times.clear()

        stats = self.get_stats()
        logger.info(
            "Datenbank geladen: '%s' | %d Songs | %d Einträge | %.2f MB",
            target, stats["num_songs"], stats["num_entries"], stats["memory_mb"],
        )

    # ------------------------------------------------------------------
    # Interne Methoden
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        """Konsolidiert alle Chunks zu sortierten NumPy-Arrays.

        Wird automatisch beim ersten lookup()-Aufruf ausgeführt.
        Danach sind insert()-Aufrufe gesperrt.

        Speicherbedarf während Finalisierung: ~2× die finale Array-Größe
        (Chunks + sortierte Arrays gleichzeitig bis Chunks freigegeben werden).
        """
        if self._finalized:
            return

        logger.info("Finalisiere Datenbank (%d Chunks) ...", len(self._chunks_hashes))

        if not self._chunks_hashes:
            self._finalized = True
            logger.info("Datenbank ist leer nach Finalisierung.")
            return

        # Alle Chunks in einem Schritt zusammenfügen
        all_hashes = np.concatenate(self._chunks_hashes)
        all_sids   = np.concatenate(self._chunks_sids)
        all_times  = np.concatenate(self._chunks_times)

        # Chunks freigeben (Speicher zurückgeben bevor Sortierung Kopien anlegt)
        self._chunks_hashes.clear()
        self._chunks_sids.clear()
        self._chunks_times.clear()

        # Stabile Sortierung nach Hash-Wert — O(N log N)
        order = np.argsort(all_hashes, kind='stable')
        self._hashes_arr   = all_hashes[order]
        self._song_ids_arr = all_sids[order]
        self._times_arr    = all_times[order]

        # Temporäre Arrays freigeben
        del all_hashes, all_sids, all_times, order

        # Anzahl einzigartiger Hashes in einem O(N) Vektoroperations-Durchlauf
        self._num_unique_hashes = _count_unique_sorted(self._hashes_arr)
        self._finalized = True

        stats = self.get_stats()
        logger.info(
            "Finalisiert: %d Einträge | %d einzigartige Hashes | %.2f MB",
            stats["num_entries"],
            stats["num_hashes"],
            stats["memory_mb"],
        )

    # ------------------------------------------------------------------
    # Dunder-Methoden
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Gibt die Anzahl einzigartiger Hash-Werte zurück (nach Finalisierung),
        bzw. die Gesamtzahl der Einträge in der Build-Phase."""
        if self._finalized:
            return self._num_unique_hashes
        return sum(len(c) for c in self._chunks_hashes)

    def __contains__(self, song_id: str) -> bool:
        """Prüft ob ein Song bereits indexiert ist."""
        return song_id in self._song_ids

    def __repr__(self) -> str:
        stats = self.get_stats()
        phase = "finalized" if self._finalized else "building"
        return (
            f"FingerprintDatabase("
            f"songs={stats['num_songs']}, "
            f"entries={stats['num_entries']}, "
            f"phase={phase})"
        )


# ------------------------------------------------------------------
# Hilfsfunktionen (modul-privat)
# ------------------------------------------------------------------

def _resolve_path(path: str | Path | None) -> Path:
    """Löst den Speicherpfad auf, immer mit .npz-Suffix."""
    if path is not None:
        p = Path(path)
    else:
        p = config.DB_PATH
    # Stelle sicher dass .npz-Suffix gesetzt ist
    if p.suffix != '.npz':
        p = p.with_suffix('.npz')
    return p


def _count_unique_sorted(arr: np.ndarray) -> int:
    """Zählt einzigartige Werte in einem sortierten Array in O(N).

    Effizienter als np.unique() da keine zweite Sortierung nötig.
    """
    if len(arr) == 0:
        return 0
    if len(arr) == 1:
        return 1
    # Vektorisiert: Anzahl der Stellen wo sich der Wert ändert + 1
    return int(np.sum(arr[1:] != arr[:-1])) + 1
