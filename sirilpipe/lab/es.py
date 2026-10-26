"""A (1+1)-CMA-ES for constrained optimisation as an Optuna sampler: local search from a feasible start under black-box constraints.

D. V. Arnold and N. Hansen, "A (1+1)-CMA-ES for Constrained Optimisation", GECCO 2012, 297-304 (Algorithm 1 and its parameter settings):
one offspring y = x + sigma A z, z ~ N(0, I), per trial; an infeasible offspring is discarded and its step, through an exponentially fading
record v_j of the steps that violated constraint j, reduces the variance of the search distribution in that direction (A <- A - beta / k
sum_j v_j w_j^T / (w_j^T w_j), w_j = A^-1 v_j); a feasible offspring replaces the parent when it is no worse, the covariance learns the
successful steps (the evolution path s, rank-one update of the Cholesky factor A, Igel, Suttorp & Hansen 2006) and the step size follows the
success rule (target success rate 2/11).  Only feasible offspring adapt the step size.

Why here: the finishing study searches ~28 settings against a dozen hard gates whose feasible region lies round the user's (Auto-finish's)
settings.  Constrained TPE over the full ranges spent 50 of 60 trials outside the gates on M 31 (2026-10-25: no feasible trial after its warm
start), while 4 of 9 small perturbations of the baseline were feasible - the situation this algorithm is for.

The search runs in the unit cube of the numeric settings (log settings in log space; integers rounded; a step that leaves the cube is reflected
into it, and the step actually taken is the one learnt).  Categorical and boolean settings, which a covariance cannot model, are mutated as in
the (1+1) evolutionary algorithm: each changes to another of its choices with probability 1 / n, n the number of settings searched
(Muehlenbein 1992), and the offspring is accepted or rejected as a whole.  The initial step size is 2 % of each range: an isotropic step's
length grows with sqrt(n), and at 5 % (a step of ~0.24 of the ranges over 23 settings) none of 10 offspring on M 31 met the gates - only
feasible offspring adapt the step size, so a start that is too wide does not recover within a budget of tens of trials.  Trials the sampler did not propose (the enqueued baseline, warm-start perturbations, results
reused for repeated settings) are taken as candidates: the parent is the best feasible trial seen.
"""
from __future__ import annotations

import math

import numpy as np
import optuna


class ConstrainedOnePlusOneES(optuna.samplers.BaseSampler):
    """Optuna sampler over ``space`` (the study's space: {name: {tune, low, high, log, choices, value}}) and the
    task's parameter definitions ``pdefs``; ``direction`` of the single objective."""

    def __init__(self, space: dict, pdefs: dict, direction: str = "minimize", seed: int = 0, sigma0: float = 0.02):
        self._fallback = optuna.samplers.RandomSampler(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.sign = 1.0 if direction == "minimize" else -1.0
        self.dims, self.cats = [], []
        for k, sp in space.items():
            pd = pdefs.get(k)
            if not pd or not sp.get("tune"):
                continue
            if pd["type"] in ("float", "int"):
                lo, hi = float(sp.get("low", pd.get("low"))), float(sp.get("high", pd.get("high")))
                lg = bool(sp.get("log", pd.get("log", False))) and lo > 0
                if hi > lo:
                    self.dims.append((k, lo, hi, lg, pd["type"] == "int"))
            elif pd["type"] == "bool":
                self.cats.append((k, [True, False]))
            elif pd["type"] == "categorical":
                ch = list(sp.get("choices") or pd["choices"])
                if len(ch) > 1:
                    self.cats.append((k, ch))
        n = max(1, len(self.dims))
        self.n = n
        # Arnold & Hansen 2012, Table 1
        self.d = 1.0 + n / 2.0
        self.c = 2.0 / (n + 2.0)
        self.c_p = 1.0 / 12.0
        self.p_target = 2.0 / 11.0
        self.c_cov = 2.0 / (n * n + 6.0)
        self.c_c = 1.0 / (n + 2.0)
        self.beta = 0.1 / (n + 2.0)
        self.sigma0 = float(sigma0)
        self.sigma = self.sigma0
        self.A = np.eye(n)
        self.s = np.zeros(n)
        self.p_succ = self.p_target
        self.v: dict[int, np.ndarray] = {}
        self.x: np.ndarray | None = None              # the parent: unit-cube numerics
        self.xc: dict = {}                            # ... and its choices
        self.fx = math.inf                            # its signed score (feasible), or its violation (while no feasible trial is known)
        self.feasible = False
        self.done: set[int] = set()
        self.proposals: dict[int, dict] = {}

    # ------------------------------------------------------------------ the unit cube
    def _to_unit(self, k, lo, hi, lg, v) -> float:
        v = float(v)
        if lg:
            return (math.log(max(v, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return (v - lo) / (hi - lo)

    def _from_unit(self, lo, hi, lg, integer, u):
        v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if lg else lo + u * (hi - lo)
        return int(round(v)) if integer else float(min(max(v, lo), hi))

    @staticmethod
    def _reflect(u: np.ndarray) -> np.ndarray:
        u = np.mod(u, 2.0)
        return np.where(u > 1.0, 2.0 - u, u)

    def _unit_of(self, params: dict, like: np.ndarray | None) -> np.ndarray:
        out = np.array(like if like is not None else np.full(self.n, 0.5), float)
        for i, (k, lo, hi, lg, _) in enumerate(self.dims):
            if k in params and isinstance(params[k], (int, float)) and not isinstance(params[k], bool):
                out[i] = min(max(self._to_unit(k, lo, hi, lg, params[k]), 0.0), 1.0)
        return out

    # ------------------------------------------------------------------ learning from finished trials
    @staticmethod
    def _violations(t) -> list[float]:
        c = t.user_attrs.get("constraints")
        if c is None:
            c = t.system_attrs.get("constraints") or [0.0]
        return [float(x) for x in c]

    def _sync(self, study):
        for t in sorted(study.trials, key=lambda q: q.number):
            if t.number in self.done or not t.state.is_finished():
                continue
            self.done.add(t.number)
            if t.state != optuna.trial.TrialState.COMPLETE or not t.values:
                continue
            g = self._violations(t)
            feas = all(x <= 0 for x in g)
            f = self.sign * float(t.values[0])
            viol = float(sum(max(x, 0.0) for x in g))
            prop = self.proposals.get(t.number)
            u = self._unit_of(t.params, prop["u"] if prop else self.x)
            choices = {k: t.params[k] for k, _ in self.cats if k in t.params}
            if self.x is None:                                       # the first result (the baseline): the parent
                self._set_parent(u, choices, f if feas else viol, feas)
                continue
            if prop is None:                                         # not proposed here: a candidate only
                if self._better(feas, f, viol):
                    self._set_parent(u, {**self.xc, **choices}, f if feas else viol, feas)
                continue
            step = prop["step"]                                      # the step taken (after reflection), in A-coordinates
            if not feas and self.feasible:
                # Arnold & Hansen: an infeasible offspring reduces the variance along the directions of the violated constraints
                az = self.A @ step
                viol_j = [j for j, x in enumerate(g) if x > 0]
                upd = np.zeros_like(self.A)
                for j in viol_j:
                    vj = self.v.get(j, np.zeros(self.n))
                    vj = (1.0 - self.c_c) * vj + self.c_c * az
                    self.v[j] = vj
                    w = np.linalg.solve(self.A, vj)
                    ww = float(w @ w)
                    if ww > 1e-300:
                        upd += np.outer(vj, w) / ww
                if viol_j:
                    self.A = self.A - (self.beta / len(viol_j)) * upd
                continue
            ok = self._better(feas, f, viol, ties=True)
            if feas:
                self.p_succ = (1.0 - self.c_p) * self.p_succ + (self.c_p if ok else 0.0)
                self.sigma *= math.exp((self.p_succ - self.p_target) / (self.d * (1.0 - self.p_target)))
                self.sigma = float(min(max(self.sigma, 1e-4), 0.5))
            if ok:
                if feas:
                    az = self.A @ step
                    self.s = (1.0 - self.c) * self.s + math.sqrt(self.c * (2.0 - self.c)) * az
                    w = np.linalg.solve(self.A, self.s)
                    ww = float(w @ w)
                    if ww > 1e-300:
                        a = math.sqrt(1.0 - self.c_cov)
                        self.A = a * self.A + (a / ww) * (math.sqrt(1.0 + self.c_cov * ww / (1.0 - self.c_cov)) - 1.0) * np.outer(self.s, w)
                self._set_parent(u, {**self.xc, **choices}, f if feas else viol, feas)

    def _better(self, feas: bool, f: float, viol: float, ties: bool = False) -> bool:
        if feas and not self.feasible:
            return True
        if not feas:
            return (not self.feasible) and (viol < self.fx or (ties and viol <= self.fx))
        return f < self.fx or (ties and f <= self.fx)

    def _set_parent(self, u, choices, fx, feas):
        self.x, self.xc, self.fx, self.feasible = np.array(u, float), dict(choices), float(fx), bool(feas)

    # ------------------------------------------------------------------ Optuna sampler interface
    def reseed_rng(self) -> None:
        self._fallback.reseed_rng()

    def infer_relative_search_space(self, study, trial):
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}

    def before_trial(self, study, trial):
        self._sync(study)
        fixed = study._storage.get_trial(trial._trial_id).system_attrs.get("fixed_params")
        if fixed or self.x is None:
            return
        z = self.rng.standard_normal(self.n)
        y = self._reflect(self.x + self.sigma * (self.A @ z))
        step = np.linalg.solve(self.A, (y - self.x) / self.sigma)
        vals = {k: self._from_unit(lo, hi, lg, it, float(y[i])) for i, (k, lo, hi, lg, it) in enumerate(self.dims)}
        pm = 1.0 / max(1, len(self.cats) + len(self.dims))        # the (1+1)-EA's rate: 1 / n over all n settings
        for k, ch in self.cats:
            cur = self.xc.get(k, ch[0])
            if self.rng.random() < pm:
                others = [c for c in ch if str(c) != str(cur)]
                cur = others[int(self.rng.integers(len(others)))] if others else cur
            vals[k] = cur
        self.proposals[trial.number] = {"u": y, "step": step, "values": vals}
        study._storage.set_trial_system_attr(trial._trial_id, "es_sigma", float(self.sigma))

    def sample_independent(self, study, trial, param_name, param_distribution):
        prop = self.proposals.get(trial.number)
        if prop is not None and param_name in prop["values"]:
            v = prop["values"][param_name]
            try:
                if param_distribution._contains(param_distribution.to_internal_repr(v)):
                    return v
            except Exception:                                   # noqa: BLE001
                pass
        return self._fallback.sample_independent(study, trial, param_name, param_distribution)

    def after_trial(self, study, trial, state, values):
        pass
