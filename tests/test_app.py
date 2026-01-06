from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_docs():
    """
    Checks if the API documentation page loads.
    (This guarantees the server is running, even if the home page is missing).
    """
    response = client.get("/docs")
    assert response.status_code == 200

def test_model_loading():
    """
    Checks if the Spacy AI model is loaded correctly.
    """
    from main import nlp
    doc = nlp("Meeting on Friday.")
    assert len(doc) > 0