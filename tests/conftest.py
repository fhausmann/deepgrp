"""Pytest config and fixtures."""
import random
import string

import pytest


@pytest.fixture
def randomword():
    "Create a random word/string."
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(20))
