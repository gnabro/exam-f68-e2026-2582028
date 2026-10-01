import pytest
from pydantic import ValidationError

from app.main import predict_endpoint, PredictionRequest

@pytest.mark.anyio
def test_predict_success_basic():
    data = PredictionRequest(features=[3.5, 1.2, 4.9])
    resp = predict_endpoint(data)
    assert resp == {"predictions": [7.0, 2.4, 9.8]}


@pytest.mark.anyio
def test_predict_success_string():
    data = PredictionRequest(features=["3.5", "1.2", "4.9"])
    resp = predict_endpoint(data)
    assert resp == {"predictions": [7.0, 2.4, 9.8]}

@pytest.mark.anyio
def test_predict_invalid_data ():
    with pytest.raises(ValidationError):
        PredictionRequest(features=["a", "b", "c"])

##AJOUT
@pytest.mark.anyio
def test_predict_success_features_vide():
    """Vérifie le comportement de predict_endpoint avec une liste de features vide."""
    data = PredictionRequest(features=[])
    result = predict_endpoint(data)
    assert result == {"predictions": []}

@pytest.mark.anyio
def test_predict_success_features_negatif():
    """Vérifie le comportement de predict_endpoint avec des valeurs négatives."""
    data = PredictionRequest(features=[-2.0, -4.5, -10.0])
    result = predict_endpoint(data)
    assert result == {"predictions": [-4.0, -9.0, -20.0]}