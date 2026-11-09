"""Import guard for an absent, disabled optional SG-DRF dependency.

The evaluated OmniSAM checkpoint has no FSMC tensors and enable_scm is False.
This module never supplies model behavior: constructing it always aborts.
"""
class DynamicFSMC:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Optional DynamicFSMC source is unavailable. This branch must remain disabled; refusing substitute computation.")
