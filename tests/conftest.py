import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def small_system(rng):
    """Random directed coupling + node frequencies for model tests."""
    N = 8
    C = rng.random((N, N)) * (rng.random((N, N)) < 0.5)
    np.fill_diagonal(C, 0)
    C *= 0.2 / C.max()
    omega = 2 * np.pi * rng.uniform(0.04, 0.07, N)
    return {"N": N, "C": C, "omega": omega, "a": -0.02, "tr": 2.0, "band": [0.008, 0.08], "beta": 0.02}
