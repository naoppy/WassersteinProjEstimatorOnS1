import numpy as np

from src.distributions.sine_skewed_vonmises import (
    sine_skewed_vonmises_periodic_cdf_analytical,
)
from src.distributions.vonmises import vonmises_periodic_cdf_numerical
from src.distributions.wrappedcauchy import wrapcauchy_periodic_cdf_analytical


def test_vonmises_cdf():
    print("Testing vonmises periodic CDF properties...")
    mu = 1.0
    kappa = 2.0
    # 1. F(0) = 0
    f0 = vonmises_periodic_cdf_numerical(np.array(0.0), mu, kappa)
    assert np.abs(f0) < 1e-12, f"vonmises F(0) is {f0}, not 0"

    # 2. F(x + 2*pi) = F(x) + 1
    xs = np.linspace(-10, 10, 100)
    f_x = vonmises_periodic_cdf_numerical(xs, mu, kappa)
    f_x_plus_2pi = vonmises_periodic_cdf_numerical(xs + 2 * np.pi, mu, kappa)
    assert np.allclose(f_x_plus_2pi, f_x + 1.0), "vonmises periodic condition failed"

    # 3. Monotonicity
    assert np.all(np.diff(f_x) >= -1e-12), "vonmises monotonicity failed"
    print("vonmises tests passed!")


def test_wrapcauchy_cdf():
    print("Testing wrapped Cauchy periodic CDF properties...")
    c = 0.5
    loc = 0.5
    scale = 1.0
    # 1. F(0) = 0
    f0 = wrapcauchy_periodic_cdf_analytical(np.array(0.0), c, loc=loc, scale=scale)
    assert np.abs(f0) < 1e-12, f"wrapped Cauchy F(0) is {f0}, not 0"

    # 2. F(x + 2*pi*scale) = F(x) + 1
    xs = np.linspace(-10, 10, 100)
    f_x = wrapcauchy_periodic_cdf_analytical(xs, c, loc=loc, scale=scale)
    f_x_plus_period = wrapcauchy_periodic_cdf_analytical(
        xs + 2 * np.pi * scale, c, loc=loc, scale=scale
    )
    assert np.allclose(f_x_plus_period, f_x + 1.0), (
        "wrapped Cauchy periodic condition failed"
    )

    # 3. Monotonicity
    assert np.all(np.diff(f_x) >= -1e-12), "wrapped Cauchy monotonicity failed"
    print("wrapped Cauchy tests passed!")


def test_sine_skewed_vonmises_cdf():
    print("Testing sine-skewed von Mises periodic CDF properties...")
    mu = 1.5
    kappa = 1.0
    lambda_ = 0.5
    # 1. F(0) = 0
    f0 = sine_skewed_vonmises_periodic_cdf_analytical(np.array(0.0), mu, kappa, lambda_)
    assert np.abs(f0) < 1e-12, f"sine-skewed von Mises F(0) is {f0}, not 0"

    # 2. F(x + 2*pi) = F(x) + 1
    xs = np.linspace(-10, 10, 100)
    f_x = sine_skewed_vonmises_periodic_cdf_analytical(xs, mu, kappa, lambda_)
    f_x_plus_2pi = sine_skewed_vonmises_periodic_cdf_analytical(
        xs + 2 * np.pi, mu, kappa, lambda_
    )
    assert np.allclose(f_x_plus_2pi, f_x + 1.0), (
        "sine-skewed von Mises periodic condition failed"
    )

    # 3. Monotonicity
    assert np.all(np.diff(f_x) >= -1e-12), "sine-skewed von Mises monotonicity failed"
    print("sine-skewed von Mises tests passed!")


if __name__ == "__main__":
    test_vonmises_cdf()
    test_wrapcauchy_cdf()
    test_sine_skewed_vonmises_cdf()
    print("All periodic CDF tests successfully verified!")
