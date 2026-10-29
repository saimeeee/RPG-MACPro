REGISTRY = {}

def _sc2_decomposer(*args, **kwargs):
    # LBF and algorithm tests do not need the StarCraft installation.
    from .sc2_decomposer import SC2Decomposer
    return SC2Decomposer(*args, **kwargs)

REGISTRY["sc2_decomposer"] = _sc2_decomposer
