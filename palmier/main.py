from fastapi import FastAPI
from palmier.api.routes import api_app

app = FastAPI()

app.mount("/", api_app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)

# Usage example
# To run the server, use the following command in your terminal:
# python palmier/main.py
