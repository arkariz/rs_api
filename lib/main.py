from typing import Union, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rs import SystemRs

app = FastAPI()

class PrediksiRequest(BaseModel):
    diagnosis: List[str]
    tindakan: List[str]
    subacute: str
    chronic: str
    sp: str
    sr: str
    si: str
    sd: str


@app.post("/prediksi")
def prediksi(prediksiRequest: PrediksiRequest):
    rs = SystemRs()
    rs.inputDiagnosisCode(prediksiRequest.diagnosis)
    rs.inputTindakanCode(prediksiRequest.tindakan)
    rs.prediksi()
    if rs.hasilPrediksi == "":
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")
    else:
        return {
            "code": 200,
            "data": {
                "prediksi": rs.hasilPrediksi,
                "jumlah": rs.jumlah
            }
        }
