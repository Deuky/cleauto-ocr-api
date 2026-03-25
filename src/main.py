from fastapi import FastAPI, UploadFile, File
from typing import Annotated
from pydantic import BaseModel
from ocr import OCR

class Prediction(BaseModel):
    file: UploadFile = File(...)

app = FastAPI()

@app.on_event("startup")
def load_model():
    OCR().init_model()

@app.post("/prediction")
def post_prediction(file: Annotated[bytes, File()]):
    return {"prediction": OCR().prediction(file)}
