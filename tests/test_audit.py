import numpy as np

from audit.analyze import bootstrap_lift


def test_lift_is_zero_for_identical_sets():
    a = np.array([0.5, 0.6, 0.55, 0.7])
    lift, lo, hi = bootstrap_lift(a, a.copy())
    assert lift == 0 and lo == 0 and hi == 0


def test_lift_sign_and_ci():
    styled = np.array([0.70, 0.72, 0.68, 0.75, 0.71])
    control = styled - 0.1
    lift, lo, hi = bootstrap_lift(styled, control)
    assert abs(lift - 0.1) < 1e-9
    assert lo <= lift <= hi


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(1)
    s, c = rng.random(8), rng.random(8)
    assert bootstrap_lift(s, c) == bootstrap_lift(s, c)
