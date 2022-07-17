import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

model = pickle.load(open("model.pkl", "rb"))
columns = ['Company', 'TypeName', 'Inches', 'Ram', 'Weight', 'Width', 'Height', 'IPS', 'TouchScreen', 'SSD', 'HDD',
           'FlashStorage', 'Hybrid', 'CpuSpeed', 'CpuCompany', 'GPU_Company', 'OS']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class Item(BaseModel):
    Company: str
    TypeName: str
    Inches: float
    Ram: int
    Weight: float
    Width: int
    Height: int
    IPS: int
    TouchScreen: int
    SSD: int
    HDD: int
    FlashStorage: int
    Hybrid: int
    CpuSpeed: float
    CpuCompany: str
    GPU_Company: str
    OS: str


@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.post('/predict')
def predict(data: Item):
    testdata = [[
        data.Company,
        data.TypeName,
        data.Inches,
        data.Ram,
        data.Weight,
        data.Width,
        data.Height,
        data.IPS,
        data.TouchScreen,
        data.SSD,
        data.HDD,
        data.FlashStorage,
        data.Hybrid,
        data.CpuSpeed,
        data.CpuCompany,
        data.GPU_Company,
        data.OS
    ]]
    y = model.predict(pd.DataFrame(columns=columns, data=testdata))
    return {'result': np.exp(y[0])}
