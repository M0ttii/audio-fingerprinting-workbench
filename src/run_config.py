"""
Central run configuration for the Audio Fingerprinting Benchmark.

Change RUN_MODE here to switch all seven notebooks between dry run and live run.
All notebooks import this module and use `cfg` to derive paths and parameters.

Usage:
    from run_config import cfg
    RUN_MODE = cfg.run_mode   # "dry" or "live"
"""

# ── SINGLE SWITCH ─────────────────────────────────────────────────────────────
RUN_MODE: str = "live"   # Change to "live" for the full benchmark run.
# ─────────────────────────────────────────────────────────────────────────────


class _Config:
    """Immutable configuration object derived from RUN_MODE."""

    def __init__(self, mode: str) -> None:
        if mode not in ("dry", "live"):
            raise ValueError(f"RUN_MODE must be 'dry' or 'live', got {mode!r}")
        self.run_mode = mode

        if mode == "dry":
            # ── NB00: Partition targets ──────────────────────────────────────
            self.n_ref   = 50    # dry_ref target (actual ≈52 after stratification)
            self.n_train = 200   # dry_train target
            self.n_ood   = 10    # dry_ood target
            self.n_query = 50    # dry mode: all ref tracks are queried
            # Partition JSON keys
            self.ref_key   = "dry_ref"
            self.train_key = "dry_train"
            self.ood_key   = "dry_ood"
            self.query_key = "dry_ref"   # NB01: entire dry_ref becomes query set

            # ── NB01: Query generation ───────────────────────────────────────
            self.query_subdir  = "dry_run"           # data/queries/dry_run/
            self.manifest_name = "manifest_dry.csv"  # data/queries/manifest_dry.csv

            # ── NB02/03: Pipeline results ────────────────────────────────────
            self.results_subdir = "dry_run"          # results/dry_run/

            # ── NB04: NeuralFP training ──────────────────────────────────────
            self.pfann_config_name  = "dry_run"
            self.checkpoints_subdir = "dry_run"      # checkpoints/dry_run/
            self.batch_size         = 32
            self.epochs             = 2
            self.pfann_list_prefix  = "dry"          # dry_train.txt, dry_ref.txt, …

            # ── NB05: Matching ───────────────────────────────────────────────
            self.score_threshold = 0.7

            # ── NB06: Plots ──────────────────────────────────────────────────
            self.plots_subdir         = "plots"          # results/plots/
            self.efficiency_tex_name  = "efficiency_table.tex"

        else:  # live
            # ── NB00: Partition targets ──────────────────────────────────────
            # Note: target sizes may exceed available data (≈17 k tracks total).
            # NB00 caps each pool to what the full partition contains:
            #   live_ref   ≈ min(8000, len(ref))      ≈ 4930
            #   live_train ≈ len(train)               ≈ 10000
            #   live_ood   ≈ min(200,  len(ood_pool))  = 200
            self.n_ref   = 8000
            self.n_train = 15000
            self.n_ood   = 200
            self.n_query = 500
            # Partition JSON keys
            self.ref_key   = "live_ref"
            self.train_key = "live_train"
            self.ood_key   = "live_ood"
            self.query_key = "live_query"  # NB01: 500-song subset of live_ref

            # ── NB01: Query generation ───────────────────────────────────────
            self.query_subdir  = "live_run"           # data/queries/live_run/
            self.manifest_name = "manifest_live.csv"  # data/queries/manifest_live.csv

            # ── NB02/03: Pipeline results ────────────────────────────────────
            self.results_subdir = "live_run"           # results/live_run/

            # ── NB04: NeuralFP training ──────────────────────────────────────
            self.pfann_config_name  = "live_run"
            self.checkpoints_subdir = "live_run"       # checkpoints/live_run/
            self.batch_size         = 64
            self.epochs             = 100
            self.pfann_list_prefix  = "live"           # live_train.txt, live_ref.txt, …

            # ── NB05: Matching ───────────────────────────────────────────────
            self.score_threshold = 0.7

            # ── NB06: Plots ──────────────────────────────────────────────────
            self.plots_subdir         = "live_run/plots"   # results/live_run/plots/
            self.efficiency_tex_name  = "efficiency_table.tex"


cfg = _Config(RUN_MODE)
