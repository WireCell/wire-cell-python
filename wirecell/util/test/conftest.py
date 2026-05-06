"""Shared pytest fixtures for wirecell.util tests."""

import pathlib
import pytest

_DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def minimal_gdml_path():
    """Return the path to the minimal synthetic GDML test fixture."""
    return _DATA / "minimal.gdml"
