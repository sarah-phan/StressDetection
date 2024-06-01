from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from MoveFile import moveFile
from ModelLoad import modelLoad
import sys

app = FastAPI()

class LoginData(BaseModel):
    username: str

@app.post('/get-username')
async def login(loginData: LoginData):
    try:
        username = loginData.username
        moveFile(username)
        return {'message': 'Data has implemented'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-label")
async def getData():
    try:
        segmented_data, labels, prediction_probability = modelLoad()
        segmented_data = segmented_data.tolist()
        labels = labels.tolist()
        prediction_probability = prediction_probability.tolist()

        # print(segmented_data)
        # print(labels)
        # print(len(segmented_data))
        # print(len(labels))

        return {'data': segmented_data, 'label': labels, 'prediction_probability': prediction_probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

