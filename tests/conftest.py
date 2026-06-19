from collections.abc import Generator
import os

os.environ["OPENAI_API_KEY"] = ""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    app = create_app(Settings(openai_api_key=None))
    with TestClient(app) as test_client:
        yield test_client
