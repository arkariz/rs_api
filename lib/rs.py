import pandas as pd
import sqlite3
import os



class SystemRs:
    database_rs = None
    diagnosis_code = ""
    tindakan_code = ""
    chronic = ""
    subacute = ""
    sp = ""
    sr = ""
    si = ""
    sd = ""
    hasilPrediksi = ""
    jumlah = ""
    tarif_rs = ""
    tarif_inacbg = ""

    def __init__(self):
        file_exists = os.path.isfile("./data/rsdb.db")
        print(file_exists)
        if not file_exists:
            self.loadExcelToDb()
            self.loadDatabse()
        else:
            self.loadDatabse()

    def loadExcelToDb(self):
        conn = sqlite3.connect("data/rsdb.db")

        df = pd.read_excel("data/data.xlsx")
        df.to_sql("rs_table", conn, if_exists="replace")
        conn.close()

    def loadDatabse(self):
        conn = sqlite3.connect("data/rsdb.db")
        df = pd.read_sql_query("SELECT * from rs_table", conn)
        conn.close()

        rs = df[
            [
                "NOKARTU",
                "KELAS_RAWAT",
                "SEX",
                "LAMA_DIRAWAT",
                "UMUR_TAHUN",
                "Diagnosis",
                "Tindakan",
                "INACBG",
                "SUBACUTE",
                "CHRONIC",
                "SP",
                "SR",
                "SI",
                "SD",
                "TARIF_INACBG",
                "TARIF_RS",
                "PROSEDUR_NON_BEDAH",
                "PROSEDUR_BEDAH",
                "KONSULTASI",
                "TENAGA_AHLI",
                "KEPERAWATAN",
                "PENUNJANG",
                "RADIOLOGI",
                "LABORATORIUM",
                "PELAYANAN_DARAH",
                "REHABILITASI",
                "KAMAR_AKOMODASI",
                "RAWAT_INTENSIF",
                "OBAT",
                "ALKES",
                "BMHP",
                "SEWA_ALAT",
                "OBAT_KRONIS",
                "OBAT_KEMO"
            ]
        ]
        print(rs.iloc[-1])

        rs.fillna("-")
        rs["Tindakan"] = rs["Tindakan"].astype(str)
        self.database_rs = rs

    def inputDiagnosisCode(self, diagnosis: list):
        input_diagnosis_list = diagnosis

        diagnosis_list = []
        for i in input_diagnosis_list:
            if i != "-":
                diagnosis_list.append(i)

        diagnosis_code = ";".join(diagnosis_list)
        if diagnosis_code == "":
            diagnosis_code = "-"

        self.diagnosis_code = str(diagnosis_code)

    def inputTindakanCode(self, tindakan: list):
        input_tindakan_list = tindakan

        tindakan_list = []
        for i in input_tindakan_list:
            if i != "-":
                tindakan_list.append(i)

        tindakan_code = ";".join(tindakan_list)
        if tindakan_code == "":
            tindakan_code = "-"

        self.tindakan_code = str(tindakan_code)
    
    def inputSpecial(self, subacute, chronic, sp, sr, si, sd):
        self.subacute = "-" if subacute == "" else subacute
        self.chronic = "-" if chronic == "" else chronic
        self.sp = "-" if sp == "" else sp
        self.sr = "-" if sr == "" else sr
        self.si = "-" if si == "" else si
        self.sd = "-" if sd == "" else sd

    def prediksi(self):
        find = self.database_rs.loc[
            (self.database_rs["Diagnosis"] == self.diagnosis_code)
            & (self.database_rs["Tindakan"] == self.tindakan_code)
            & (self.database_rs["SUBACUTE"] == self.subacute)
            & (self.database_rs["CHRONIC"] == self.chronic)
            & (self.database_rs["SP"] == self.sp)
            & (self.database_rs["SR"] == self.sr)
            & (self.database_rs["SI"] == self.si)
            & (self.database_rs["SD"] == self.sd)
        ]

        if find.empty == False:
            tarif_inacbg = find["TARIF_INACBG"].iloc[0]
            tarif_rs = find["TARIF_RS"].mean()
            prediksi = ""
            jumlah = 0

            if tarif_inacbg < tarif_rs:
                prediksi = "rugi"
                jumlah = tarif_inacbg - tarif_rs
            else:
                prediksi = "untung"
                jumlah = tarif_inacbg - tarif_rs

            self.hasilPrediksi = prediksi
            self.jumlah = str(jumlah)
            self.tarif_rs = str(tarif_rs)
            self.tarif_inacbg = str(tarif_inacbg)

    def inputData(self, data):
        sql = '''INSERT INTO rs_table(
                NOKARTU,
                KELAS_RAWAT,
                SEX,
                LAMA_DIRAWAT,
                UMUR_TAHUN,
                Diagnosis,
                Tindakan,
                INACBG,
                SUBACUTE,
                CHRONIC,
                SP,
                SR,
                SI,
                SD,
                TARIF_INACBG,
                TARIF_RS,
                PROSEDUR_NON_BEDAH,
                PROSEDUR_BEDAH,
                KONSULTASI,
                TENAGA_AHLI,
                KEPERAWATAN,
                PENUNJANG,
                RADIOLOGI,
                LABORATORIUM,
                PELAYANAN_DARAH,
                REHABILITASI,
                KAMAR_AKOMODASI,
                RAWAT_INTENSIF,
                OBAT,
                ALKES,
                BMHP,
                SEWA_ALAT,
                OBAT_KRONIS,
                OBAT_KEMO) 
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
        conn = sqlite3.connect("data/rsdb.db")
        cursor = conn.cursor()
        cursor.execute(sql, data)
        conn.commit()
        conn.close()