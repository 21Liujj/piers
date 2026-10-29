"""Experiment registry: E0/T0/T1/T2/U1/B1/M/S/C + negative controls.

Canonical suite codes are short (E0, T0, ...). Configs must set
  suite: <short code>
  run_id: <descriptive id, e.g. T0-grounded-direct>
so scheduler deps resolve regardless of run_id naming.
"""
SUITES = {
    "E0": "effective_start UDA baseline (frozen effective weights)",
    "T0": "grounded_direct full-ERP text backend",
    "T1": "grounded_fov fixed-FoV + remap + assoc, no new modules",
    "T2": "T1 + semantic_to_text (E0 cache)",
    "U1": "text_to_pseudo adaptation + matched no-text control",
    "B1": "bidirectional with declared teacher/cache rounds",
    "M": "memory ablations at matched token budgets",
    "S": "seam ablations at matched extra-window budgets",
    "C": "combo (after singles pass)",
    "NEG": "negative controls: no-guide, mismatched semantics, shuffled query",
}

# dependency order for scheduler.
# Maps a canonical suite code -> the CANONICAL CODES of its dependencies.
DEPS = {"T2": ["E0"], "U1": ["T1"], "B1": ["T2", "U1"], "C": ["B1", "M", "S"]}

# The run_id each canonical suite is expected to write under. Dependency
# checks look in these directories, so they must match the configs'
# `run_id:` fields exactly (see configs/piers/*.yaml).
SUITE_RUN_IDS = {
    "E0": "E0-effective-start",
    "T0": "T0-grounded-direct",
    "T1": "T1-grounded-fov",
    "T2": "T2-semantic-to-text",
    "U1": "U1-text-to-pseudo",
    "B1": "B1-bidirectional",
    "M": "M-memory",
    "S": "S-seam",
    "C": "C-combo",
    "NEG": "NEG-controls",
}


# allow long "T2-semantic-to-text" style names in configs: canonicalize by prefix
def canonical_suite(name: str) -> str:
    if not name:
        return name
    head = str(name).split("-")[0].split("_")[0].strip().upper()
    return head if head in SUITES else str(name)


def run_id_for(suite: str) -> str:
    """Directory name a canonical suite writes its results under."""
    return SUITE_RUN_IDS.get(canonical_suite(suite), canonical_suite(suite))

MEMORY_VARIANTS = ["current", "nocompress_ref", "uniform_budget", "topk_fixed",
                   "pos_dedup", "pos_dedup_priority", "neighborhood"]
SEAM_VARIANTS = ["linear", "fixed_scm", "uniform_extra", "adaptive_extra"]
COLLAB_MODES = ["none", "semantic_to_text", "text_to_pseudo", "bidirectional"]
