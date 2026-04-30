from fastapi import FastAPI

from lightrag_enterprise.little_bull.router import create_little_bull_router


def test_little_bull_popular_labels_limit_matches_frontend_contract():
    app = FastAPI()
    app.include_router(create_little_bull_router(rag=object(), doc_manager=object()))

    openapi = app.openapi()
    parameters = openapi["paths"]["/little-bull/graph/label/popular"]["get"]["parameters"]
    limit = next(parameter for parameter in parameters if parameter["name"] == "limit")

    assert limit["schema"]["default"] == 300
    assert limit["schema"]["maximum"] >= 300
