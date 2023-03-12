from typing import Union, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rs import SystemRs
import keras
import pandas as pd
import math
import pickle




app = FastAPI()

class LamaRawatRequest(BaseModel):
    diagnosis: List[str]
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
    

@app.post("/prediksi-lama-rawat")
def prediksiLamaRawat(lamaRawatRequest: LamaRawatRequest):
    with open('data/predictorScaler.pkl', 'rb') as file:
        predictorScaler = pickle.load(file)
    
    with open('data/targerScaler.pkl', 'rb') as file:
        targerScaler = pickle.load(file)

    model = keras.models.load_model('data/model.h5')
    diag_list = pd.read_excel('data/data.xlsx')
    diag_list = diag_list[['Diagnosis']]

    diag_list['DiagnosisCAT'] = diag_list
    diag_list['DiagnosisTrans'] = diag_list['DiagnosisCAT'].astype('category')
    diag_list['DiagnosisTrans'] = diag_list['DiagnosisTrans'].cat.reorder_categories(diag_list['DiagnosisCAT'].unique(), ordered=True)
    diag_list['DiagnosisTrans'] = diag_list['DiagnosisTrans'].cat.codes

    input_diagnosis_list = lamaRawatRequest.diagnosis

    diagnosis_list = []
    for i in input_diagnosis_list:
        if i != "-":
            diagnosis_list.append(i)

    diagnosis_code = ";".join(diagnosis_list)
    if diagnosis_code == "":
        diagnosis_code = "-"

    diagnosis_code = str(diagnosis_code)

    index = diag_list.index[diag_list['Diagnosis'] == diagnosis_code].to_list()
    if index.__len__() !=0:
        requestDiagnosis = diag_list.iloc[index[0]]

        data = [[requestDiagnosis['DiagnosisTrans'], lamaRawatRequest.sex, lamaRawatRequest.umur]]
        data = pd.DataFrame(data, columns=['Diagnosis', 'SEX', 'UMUR_TAHUN'])

        X = data[['Diagnosis', 'SEX', 'UMUR_TAHUN']].values
        X = predictorScaler.transform(X)

        prediction = model.predict(X)
        prediction = targerScaler.inverse_transform(prediction)
        prediction = prediction[0]
        prediction = math.ceil(prediction)
        return {
            "code": 200,
            "data": {
                "prediksi": prediction
            }
        }
    else:
        raise HTTPException(status_code=401, detail="Data tidak ditemukan")


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

