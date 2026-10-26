"""Experiment registry: E0/T0/T1/T2/U1/B1/M/S/C + negative controls."""
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

# dependency order for scheduler
DEPS = {"T2": ["E0"], "U1": ["T1"], "B1": ["T2", "U1"], "C": ["B1", "M", "S"]}

MEMORY_VARIANTS = ["current", "nocompress_ref", "uniform_budget", "topk_fixed",
                   "pos_dedup", "pos_dedup_priority", "neighborhood"]
SEAM_VARIANTS = ["linear", "fixed_scm", "uniform_extra", "adaptive_extra"]
COLLAB_MODES = ["none", "semantic_to_text", "text_to_pseudo", "bidirectional"]
