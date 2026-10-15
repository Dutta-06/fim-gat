"""Damped prices for the improvement phase: the "thermostat" of docs/design_v2.md section 6.

Plain dual ascent with a best-response seed player oscillates, so the prices here only steer
proposals (the verifier decides acceptance) and are damped:

* spread price mu: PI control on the relative floor error e = ((1 + setpoint) floor - total) / floor,
  with an integral clipped at 0 (anti-windup), so mu rises while the total is below the setpoint and
  falls back to 0 once it holds comfortably. With setpoint = 0 (default) and a verifier that keeps
  every state at or above the floor, e <= 0 throughout, so mu stays 0 (docs/theory_v2.md, W4); a
  positive setpoint makes the price act before the floor binds;
* group prices: alpha over the groups above the size-weighted centre rate and beta over those
  below, each in proportion to its distance from the centre, scaled by D / (D + tau) so the push
  fades as the disparity D closes.

Signed weights for scoring follow objective.price_weights(alpha, beta, mu, sizes, present).
"""
import numpy as np

from .config import PRICE_TAU
from .objective import disparity, present_mask, to_numpy


def damped_prices(rates, present, tau=PRICE_TAU, sizes=None):
    """(alpha, beta), numpy (G,), each summing to D / (D + tau) over present groups, 0 elsewhere.

    The centre is the size-weighted mean rate of present groups; without sizes it is the plain mean.
    """
    r = to_numpy(rates)
    p = present_mask(present, len(r))
    alpha, beta = np.zeros(len(r)), np.zeros(len(r))
    D = disparity(r, p)
    if D <= 0.0:
        return alpha, beta
    rp = r[p]
    wts = np.ones(rp.size) if sizes is None else to_numpy(sizes)[p]
    if wts.sum() <= 0:
        wts = np.ones(rp.size)
    dev = rp - (wts * rp).sum() / wts.sum()
    s = D / (D + tau)
    up, dn = np.maximum(dev, 0.0), np.maximum(-dev, 0.0)
    if up.sum() > 0:
        alpha[p] = s * up / up.sum()
    if dn.sum() > 0:
        beta[p] = s * dn / dn.sum()
    return alpha, beta


class PIController:
    """Spread price mu by PI control on the floor error, plus damped two-sided group prices.

    sizes (G,) sets the size-weighted centre; pass it here or to update (e.g. CoverState.sizes).
    setpoint: relative slack above the floor that the spread price aims for (0 = the floor itself).
    """

    def __init__(self, kp=1.0, ki=0.5, tau=PRICE_TAU, sizes=None, setpoint=0.0):
        self.kp, self.ki, self.tau = float(kp), float(ki), float(tau)
        self.setpoint = float(setpoint)
        self.sizes = None if sizes is None else to_numpy(sizes)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.mu = 0.0
        self.alpha = self.beta = None

    def update(self, rates, present, total, floor, sizes=None):
        """One control step. Returns (alpha, beta, mu); also kept as attributes."""
        floor, total = float(floor), float(total)
        e = ((1.0 + self.setpoint) * floor - total) / max(abs(floor), 1e-12)
        self.integral = max(0.0, self.integral + e)
        self.mu = max(0.0, self.kp * e + self.ki * self.integral)
        # the contract's update() has no sizes; without them (here or in __init__) the centre is unweighted
        self.alpha, self.beta = damped_prices(rates, present, self.tau, self.sizes if sizes is None else sizes)
        return self.alpha, self.beta, self.mu
