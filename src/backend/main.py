import pickle
from typing import List
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelRequest(BaseModel):
    input_args: List[float]

class ModelResponse(BaseModel):
    result: List[float]

with open('models/regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.post("/model")
async def process_input(req: ModelRequest):
    return ModelResponse(result=list(model.predict(np.array(req.input_args).reshape(-1, 1))))
    # res.message = "Ehehehe"
    # res.param = model.predict(np.array(req.param).reshape(-1, 1))
    # return res
