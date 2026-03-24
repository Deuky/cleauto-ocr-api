from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

class Prediction(BaseModel):
	image: UploadFile = File(...)

app = FastAPI()

@app.post("/prediction")
def post_prediction(prediction: Prediction):
    return {"Hello": Prediction.image.filename}