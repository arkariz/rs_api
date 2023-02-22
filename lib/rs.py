import pandas as pd
import sqlite3
import os



class SystemRs:
    database_rs = None
    diagnosis_code = ""
    tindakan_code = ""
    hasilPrediksi = ""
    jumlah = 0

    def __init__(self):
        file_exists = os.path.exists("data/rsdb.db")
        print(file_exists)
        if not file_exists:
            self.loadExcelToDb()
            self.loadDatabse()
        else:
            self.loadDatabse()

    def loadExcelToDb(self):
        conn = sqlite3.connect("data/rsdb.db")

        df = pd.read_excel("data/data.xlsx")
        df.to_sql("data_rs", conn, if_exists="replace")
        conn.execute(
            """
            create table rs_table as
            select * from data_rs
            """
        )
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
                "lama dirawat",
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
            ]
        ]

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

        self.diagnosis_code = diagnosis_code

    def inputTindakanCode(self, tindakan: list):
        input_tindakan_list = tindakan

        tindakan_list = []
        for i in input_tindakan_list:
            if i != "-":
                tindakan_list.append(i)

        tindakan_code = ";".join(tindakan_list)
        if tindakan_code == "":
            tindakan_code = "-"

        self.tindakan_code = tindakan_code

    def prediksi(self):
        find = self.database_rs.loc[
            (self.database_rs["Diagnosis"] == self.diagnosis_code)
            & (self.database_rs["Tindakan"] == self.tindakan_code)
        ]

        if find.empty == False:
            tarif_inacbg = find["TARIF_INACBG"].iloc[0]
            tarif_rs = find["TARIF_RS"].sum()
            prediksi = ""
            jumlah = 0

            if tarif_rs > tarif_inacbg:
                prediksi = "untung"
                jumlah = tarif_rs - tarif_inacbg
            else:
                prediksi = "rugi"
                jumlah = tarif_rs - tarif_inacbg

            self.hasilPrediksi = prediksi
            self.jumlah = str(jumlah)
            print(self.hasilPrediksi)
            print(self.jumlah)
