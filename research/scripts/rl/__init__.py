"""Reinforcement-learning tuner for the evolvable AIs (default: Evo4).

This package is a sibling of :mod:`scripts.evolve`. Same inputs (one of
the four :class:`AIProfile` dataclasses from :mod:`scripts.evolve.profiles`,
one of the eight opponent modes from :mod:`scripts.evolve.opponents`),
same output shape (a JSON weights file + a history plot under
``artifacts/``), but a fundamentally different optimizer:

* GA       — population, tournament selection, crossover, mutation,
              elitism. Discrete parent-picking.
* **RL/ES** — parameter-space policy-gradient via **Evolution Strategies**
              (Salimans et al., 2017, *"Evolution Strategies as a
              Scalable Alternative to Reinforcement Learning"*). A single
              point ``θ`` moves via mirrored-sampling gradient estimates
              and an Adam update.

The ES interpretation is standard RL: each candidate AI is a policy
(``bid = bias + Σ wᵢ·featureᵢ`` for each head, clamped to the legal
bid range), the game is a Markov decision process with an episodic
score-margin return, and the outer loop performs a stochastic-gradient
update on the policy parameters. ES estimates the policy gradient in
parameter space rather than in action space — which is the right call
for Evo4 because its bid is deterministic given ``θ``, the return is
purely episodic, and the "action space" (integer bids clamped to a
per-state cap) is awkward for canonical REINFORCE.

Entry point::

    python -m scripts.rl --ai evo4 --opponent vs_all

See :func:`scripts.rl.trainer.run_es` for the algorithmic core.
"""

from .trainer import ESResult, run_es

__all__ = ["ESResult", "run_es"]
