from collections.abc import Generator
import os
from pathlib import Path

os.environ["OPENAI_API_KEY"] = ""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


@pytest.fixture()
def client(tmp_path: Path) -> Generator[TestClient, None, None]:
    app = create_app(Settings(openai_api_key=None, sqlite_path=str(tmp_path / "rag.db")))
    with TestClient(app) as test_client:
        yield test_client