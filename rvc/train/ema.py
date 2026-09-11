import contextlib

import torch


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class WeightEMA:
    """
    Exponential moving average of a model's weights.

    Until the run is longer than the averaging horizon, the plain running mean
    is used instead, so the average does not start biased toward the initial
    weights.

    Args:
        model (torch.nn.Module): Model whose weights are averaged.
        decay (float, optional): Decay of the average. Defaults to 0.999.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.updates = 0

        state = _unwrap(model).state_dict()
        self.shadow = {key: value.detach().clone() for key, value in state.items()}

        # ``state_dict`` tensors share storage with the parameters, and
        # ``load_state_dict`` copies into them, so these stay valid.
        self._live = [v for v in state.values() if v.is_floating_point()]
        self._averaged = [self.shadow[k] for k, v in state.items() if v.is_floating_point()]
        self._other_keys = [k for k, v in state.items() if not v.is_floating_point()]

    @torch.no_grad()
    def update(self, model):
        """
        Move the average toward the model's current weights.

        Args:
            model (torch.nn.Module): Model holding the live training weights.
        """
        self.updates += 1
        decay = min(self.decay, 1.0 - 1.0 / self.updates)
        torch._foreach_lerp_(self._averaged, self._live, 1.0 - decay)
        if self._other_keys:
            state = _unwrap(model).state_dict()
            for key in self._other_keys:
                self.shadow[key].copy_(state[key])

    @contextlib.contextmanager
    def applied(self, model):
        """
        Temporarily load the averaged weights into the model.

        Args:
            model (torch.nn.Module): Model to load the averaged weights into.
        """
        module = _unwrap(model)
        backup = {
            key: value.detach().to("cpu", copy=True)
            for key, value in module.state_dict().items()
        }
        module.load_state_dict(self.shadow, strict=True)
        try:
            yield module
        finally:
            module.load_state_dict(backup, strict=True)
