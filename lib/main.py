from typing import Union, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lib.rs import SystemRs

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
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")
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

