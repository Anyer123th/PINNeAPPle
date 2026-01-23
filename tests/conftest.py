import os
import pytest

@pytest.fixture(scope="session")
def rng_seed():
    return int(os.environ.get("PINNEAPLE_TEST_SEED", "1234"))
