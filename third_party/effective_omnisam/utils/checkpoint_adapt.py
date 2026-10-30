import torch
from typing import Dict, Tuple


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", ""): v for k, v in state.items()}


def _find_linear_pred_keys(state: Dict[str, torch.Tensor]) -> Tuple[str, str]:
    """
    Return (weight_key, bias_key) for the decoder classifier conv.
    Supports both OmniSAM and OmniSAM_adapt.
    """
    candidates = [
        ("decoder.linear_pred.weight", "decoder.linear_pred.bias"),
        ("decoder.linear_pred.weight", None),
    ]
    for w, b in candidates:
        if w in state and (b is None or b in state):
            return w, b
    raise KeyError("Cannot find decoder.linear_pred.* in checkpoint state dict.")


def expand_decoder_13_to_19(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Expand a 13-class checkpoint classifier weights to 19-class Cityscapes TrainIds space.

    Mapping:
    - shared classes copy 1:1 by name order used in this repo (13-class list).
    - rider initialized from person
    - truck/bus/train/motorcycle/bicycle initialized from car
    """
    state = dict(state)
    w_key, b_key = _find_linear_pred_keys(state)
    w13 = state[w_key]
    b13 = state[b_key] if b_key is not None else None

    if w13.shape[0] != 13:
        raise ValueError(f"Expected 13-class weights, got shape={tuple(w13.shape)}")

    w19 = w13.new_zeros((19, *w13.shape[1:]))
    b19 = b13.new_zeros((19,)) if b13 is not None else None

    # 13-class order in this repo:
    # 0 road,1 sidewalk,2 building,3 wall,4 fence,5 pole,6 tlight,7 tsign,
    # 8 veg,9 terrain,10 sky,11 person,12 car
    shared_13_to_19 = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,  # person -> person
        12: 13,  # car -> car
    }

    for src_i, dst_i in shared_13_to_19.items():
        w19[dst_i].copy_(w13[src_i])
        if b19 is not None and b13 is not None:
            b19[dst_i].copy_(b13[src_i])

    # rider from person
    w19[12].copy_(w13[11])
    if b19 is not None and b13 is not None:
        b19[12].copy_(b13[11])

    # truck/bus/train/motorcycle/bicycle from car
    for dst_i in [14, 15, 16, 17, 18]:
        w19[dst_i].copy_(w13[12])
        if b19 is not None and b13 is not None:
            b19[dst_i].copy_(b13[12])

    state[w_key] = w19
    if b_key is not None and b19 is not None:
        state[b_key] = b19
    return state


def smart_load_checkpoint(model, ckpt_path: str):
    """
    Load checkpoint with automatic 13->19 head expansion when needed.
    """
    if not ckpt_path:
        return
    raw = torch.load(ckpt_path, map_location="cpu")
    raw = _strip_module_prefix(raw)

    model_state = model.state_dict()
    # if model expects 19 but ckpt provides 13 for decoder head, expand it
    try:
        w_key, b_key = _find_linear_pred_keys(raw)
    except KeyError:
        model.load_state_dict(raw, strict=False)
        return

    if w_key in model_state:
        want = model_state[w_key].shape[0]
        got = raw[w_key].shape[0]
        if want == 19 and got == 13:
            raw = expand_decoder_13_to_19(raw)

    model.load_state_dict(raw, strict=False)

