"""Stdlib Adam optimizer used by the ES trainer.

Evolution Strategies in parameter space needs an outer-loop optimizer
that eats a noisy gradient estimate and spits out a stable update. Plain
SGD works, but Adam is the default in every RL-ES reference
implementation (Salimans et al. 2017, the `estool`/`evostra` libraries)
because the per-parameter learning rate reduces sensitivity to the
noise scale ``σ``.

This is a direct implementation of Kingma & Ba (2014) with bias
correction, written against the Python stdlib so the RL package has
no numpy dependency. Fast enough for a 35-vector; for 10⁶-parameter
networks this would be a bottleneck, but our "policy" is 35 floats.

Invariants:

* ``step`` does NOT mutate its input ``theta`` list — it returns a new
  list. The trainer relies on that to keep a clean copy of the
  pre-update parameters available during the same iteration (for
  logging, resume-state snapshots, etc).
* ``state_dict`` / ``from_state`` round-trips all mutable state
  (``m``, ``v``, ``t``, ``lr``, ``beta1``, ``beta2``, ``eps``) so a
  resume reproduces the same sequence of updates bit-for-bit given
  the same gradient stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Adam:
    """Adam optimizer over a flat list[float] parameter vector.

    Standard Adam hyperparameters. Default ``lr=0.03`` is what the ES
    trainer uses — larger than the neural-net default ``1e-3`` because
    the parameters here are bounded in ``[-5, 5]`` and the gradient
    estimate is sample-limited to tens of games, so bigger step sizes
    dominate the per-step noise.
    """

    lr: float = 0.03
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: list[float] = field(default_factory=list)
    v: list[float] = field(default_factory=list)
    t: int = 0

    def init(self, n: int) -> None:
        """Allocate the moment buffers for a ``n``-dimensional parameter.

        Idempotent: calling ``init(n)`` twice with the same ``n`` is a
        no-op (``m``/``v`` keep their current values). Calling with a
        different ``n`` resets the buffers.
        """
        if len(self.m) != n or len(self.v) != n:
            self.m = [0.0] * n
            self.v = [0.0] * n
            self.t = 0

    def step(self, theta: list[float], grad: list[float]) -> list[float]:
        """Return ``theta + update`` as a fresh list. Mutates internal state.

        Ascent direction: we are *maximising* the ES objective (reward),
        so the step is ``θ + α·m̂/(√v̂ + ε)`` rather than the usual
        ``θ - α·…`` used for loss minimisation.

        The input ``theta`` and ``grad`` are not mutated.
        """
        if len(theta) != len(grad):
            raise ValueError(
                f"Adam.step: theta has {len(theta)} params, "
                f"grad has {len(grad)}"
            )
        if not self.m:
            self.init(len(theta))
        self.t += 1
        b1 = self.beta1
        b2 = self.beta2
        lr = self.lr
        eps = self.eps
        # Bias-correction factors applied to the learning rate so we
        # don't have to scale m̂ and v̂ separately. This is the common
        # "efficient form" of Adam from section 2 of the paper.
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        lr_t = lr * (bc2 ** 0.5) / bc1
        out: list[float] = [0.0] * len(theta)
        for i, (w, g) in enumerate(zip(theta, grad)):
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g
            update = lr_t * self.m[i] / ((self.v[i] ** 0.5) + eps)
            out[i] = w + update
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "m": list(self.m),
            "v": list(self.v),
            "t": self.t,
        }

    @classmethod
    def from_state(cls, d: dict[str, Any]) -> "Adam":
        return cls(
            lr=d["lr"],
            beta1=d["beta1"],
            beta2=d["beta2"],
            eps=d["eps"],
            m=list(d["m"]),
            v=list(d["v"]),
            t=int(d["t"]),
        )
