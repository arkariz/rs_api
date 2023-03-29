from typing import Union, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rs import SystemRs
from nn import NeuralNetwork
import keras
import pandas as pd
import math
import pickle
import uvicorn
from sklearn.preprocessing import StandardScaler
import sqlite3
import random




app = FastAPI()

class LamaRawatRequest(BaseModel):
    diagnosis: List[str]
    tindakan: List[str]
    umur: int
    sex: int

class PrediksiRequest(BaseModel):
    diagnosis: List[str]
    tindakan: List[str]
    subacute: str
    chronic: str
    sp: str
    sr: str
    si: str
    sd: str

class InputDataRequest(BaseModel):
    NOKARTU: str
    KELAS_RAWAT: int
    SEX: int
    lamadirawat: int
    UMUR_TAHUN: int
    diagnosis: str
    tindakan: str
    INACBG: str
    subacute: str
    chronic: str
    sp: str
    sr: str
    si: str
    sd: str
    TARIF_INACBG: int
    TARIF_RS: int
    PROSEDUR_NON_BEDAH: int
    PROSEDUR_BEDAH: int
    KONSULTASI: int
    TENAGA_AHLI: int
    KEPERAWATAN: int
    PENUNJANG: int
    RADIOLOGI: int
    LABORATORIUM: int
    PELAYANAN_DARAH: int
    REHABILITASI: int
    KAMAR_AKOMODASI: int
    RAWAT_INTENSIF: int
    OBAT: int
    ALKES: int
    BMHP: int
    SEWA_ALAT: int
    OBAT_KRONIS: int
    OBAT_KEMO: int
    
@app.get("/create-model")
def createModel():
    nn = NeuralNetwork()
    nn.createModel()
    return {
        "code": 200,
        "data": {
            "message": "ok"
        }
    }

@app.post("/prediksi-lama-rawat")
def prediksiLamaRawat(lamaRawatRequest: LamaRawatRequest):
    StandardScaler()
    with open('data/predictorScaler.pkl', 'rb') as file:
        predictorScaler = pickle.load(file)
    
    with open('data/targerScaler.pkl', 'rb') as file:
        targerScaler = pickle.load(file)

    conn = sqlite3.connect("data/rsdb.db")
    df = pd.read_sql_query("SELECT * from rs_table", conn)
    conn.close()

    model = keras.models.load_model('data/model.h5')

    df['DiagnosisCAT'] = df['Diagnosis']
    df['DiagnosisTrans'] = df['DiagnosisCAT'].astype('category')
    df['DiagnosisTrans'] = df['DiagnosisTrans'].cat.reorder_categories(df['DiagnosisCAT'].unique(), ordered=True)
    df['DiagnosisTrans'] = df['DiagnosisTrans'].cat.codes

    df['TindakanCAT'] = df['Tindakan']
    df['TindakanTrans'] = df['TindakanCAT'].astype('category')
    df['TindakanTrans'] = df['TindakanTrans'].cat.reorder_categories(df['TindakanCAT'].unique(), ordered=True)
    df['TindakanTrans'] = df['TindakanTrans'].cat.codes

    input_diagnosis_list = lamaRawatRequest.diagnosis

    diagnosis_list = []
    for i in input_diagnosis_list:
        if i != "-":
            diagnosis_list.append(i)

    diagnosis_code = ";".join(diagnosis_list)
    if diagnosis_code == "":
        diagnosis_code = "-"

    diagnosis_code = str(diagnosis_code)

    input_tindakan_list = lamaRawatRequest.tindakan

    tindakan_list = []
    for i in input_tindakan_list:
        if i != "-":
            tindakan_list.append(i)

    tindakan_code = ";".join(tindakan_list)
    if tindakan_code == "":
        tindakan_code = "-"

    tindakan_code = str(tindakan_code)

    is_diagnosis_exist = df.loc[df['Diagnosis'] == diagnosis_code]['DiagnosisTrans'].to_list()
    is_tindakan_exist = df.loc[df['Tindakan'] == tindakan_code]['TindakanTrans'].to_list()
    if not is_diagnosis_exist:
        raise HTTPException(status_code=401, detail="Data tidak ditemukan")
    
    if not is_tindakan_exist:
        raise HTTPException(status_code=401, detail="Data tidak ditemukan")

    data = [[is_diagnosis_exist[0], is_tindakan_exist[0] ,lamaRawatRequest.sex, lamaRawatRequest.umur]]
    data = pd.DataFrame(data, columns=['Diagnosis', 'Tindakan' ,'SEX', 'UMUR_TAHUN'])

    X = data[['Diagnosis', 'Tindakan', 'SEX', 'UMUR_TAHUN']].values
    X = predictorScaler.transform(X)

    prediction = model.predict(X)
    prediction = targerScaler.inverse_transform(prediction)
    prediction = prediction[0]
    
    if random.randint(0, 1) == 0:
        prediction = math.ceil(prediction)
    else:
        prediction = math.floor(prediction)

    return {
        "code": 200,
        "data": {
            "prediksi": prediction
        }
    }


@app.post("/prediksi")
def prediksi(prediksiRequest: PrediksiRequest):
    rs = SystemRs()
    rs.inputDiagnosisCode(prediksiRequest.diagnosis)
    rs.inputTindakanCode(prediksiRequest.tindakan)
    rs.inputSpecial(prediksiRequest.subacute,
                    prediksiRequest.chronic, 
                    prediksiRequest.sp, 
                    prediksiRequest.sr, 
                    prediksiRequest.si, 
                    prediksiRequest.sd)
    rs.prediksi()
    if rs.hasilPrediksi == "":
        raise HTTPException(status_code=401, detail="Data tidak ditemukan")
    else:
        return {
            "code": 200,
            "data": {
                "prediksi": rs.hasilPrediksi,
                "tarif_rs": rs.tarif_rs,
                "tarif_inacbg": rs.tarif_inacbg,
                "jumlah": rs.jumlah
            }
        }

@app.post("/input-data")
def inputData(inputData: InputDataRequest):
    rs = SystemRs()
    data = [
            inputData.NOKARTU,
            inputData.KELAS_RAWAT,
            inputData.SEX,
            inputData.lamadirawat,
            inputData.UMUR_TAHUN,
            inputData.diagnosis,
            inputData.tindakan,
            inputData.INACBG,
            inputData.subacute,
            inputData.chronic,
            inputData.sp,
            inputData.sr,
            inputData.si,
            inputData.sd,
            inputData.TARIF_INACBG,
            inputData.TARIF_RS,
            inputData.PROSEDUR_NON_BEDAH,
            inputData.PROSEDUR_BEDAH,
            inputData.KONSULTASI,
            inputData.TENAGA_AHLI,
            inputData.KEPERAWATAN,
            inputData.PENUNJANG,
            inputData.RADIOLOGI,
            inputData.LABORATORIUM,
            inputData.PELAYANAN_DARAH,
            inputData.REHABILITASI,
            inputData.KAMAR_AKOMODASI,
            inputData.RAWAT_INTENSIF,
            inputData.OBAT,
            inputData.ALKES,
            inputData.BMHP,
            inputData.SEWA_ALAT,
            inputData.OBAT_KRONIS,
            inputData.OBAT_KEMO
        ]
    rs.inputData(data=data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)