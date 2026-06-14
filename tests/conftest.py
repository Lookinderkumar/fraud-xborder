"""
tests/conftest.py
Pytest configuration for the test suite.
"""
import pytest

# Restrict anyio to asyncio backend only (trio is not installed)
@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param
