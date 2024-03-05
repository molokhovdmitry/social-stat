from fastapi.testclient import TestClient
from src.main import app
import pandas as pd


client = TestClient(app)


def test_home():
    """Test home page."""
    response = client.get("/")
    assert response.status_code == 200


def test_predict():
    """Test predict method on an example video."""
    TEST_VIDEO_ID = "0peXnOnDgQ8"
    response = client.get(
        "/predict/",
        params={"video_id": TEST_VIDEO_ID}
    )
    df = pd.read_json(response, orient='records')

    # Ensure the DataFrame has the right amount of columns
    assert df.shape[1] == 39
    # Ensure there are no NaN values
    assert df.isna().sum().sum() == 0
