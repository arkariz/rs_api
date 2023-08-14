# General data analysis/plotting
import os
import numpy as np
import pandas as pd
import pickle
import sqlite3

# Data preprocessing
from sklearn.preprocessing import StandardScaler

# Neural Net modules
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class NeuralNetwork:
    def createModel(self):
        conn = sqlite3.connect("data/rsdb.db")
        df = pd.read_sql_query("SELECT * from rs_table", conn)
        conn.close()

        df = df[['Diagnosis', 'Tindakan', 'SEX', 'UMUR_TAHUN', 'LAMA_DIRAWAT', 'INACBG', 'HARI']]

        df['DiagnosisCAT'] = df['Diagnosis']
        df['Diagnosis'] = df['DiagnosisCAT'].astype('category')
        df['Diagnosis'] = df['Diagnosis'].cat.reorder_categories(df['DiagnosisCAT'].unique(), ordered=True)
        df['Diagnosis'] = df['Diagnosis'].cat.codes

        df['TindakanCAT'] = df['Tindakan']
        df['Tindakan'] = df['TindakanCAT'].astype('category')
        df['Tindakan'] = df['Tindakan'].cat.reorder_categories(df['TindakanCAT'].unique(), ordered=True)
        df['Tindakan'] = df['Tindakan'].cat.codes

        df['HARICAT'] = df['HARI']
        df['HARI'] = df['HARICAT'].astype('category')
        df['HARI'] = df['HARI'].cat.reorder_categories(df['HARICAT'].unique(), ordered=True)
        df['HARI'] = df['HARI'].cat.codes

        severity = []
        for index, row in df.iterrows():
            if row['INACBG'].split("-")[-1] == "I":
                severity.append(1)
            elif row['INACBG'].split("-")[-1] == "II":
                severity.append(2)
            else:
                severity.append(3)

        df['Severity'] = severity

        df.dropna(axis=0, inplace=True)

        X = df[['Diagnosis', 'Tindakan', 'SEX', 'UMUR_TAHUN', 'Severity', 'HARI']].values
        y = df[['LAMA_DIRAWAT']].values
        
        ### Sandardization of data ###
        PredictorScaler=StandardScaler()
        TargetVarScaler=StandardScaler()
        
        # # Storing the fit object for later reference
        PredictorScalerFit=PredictorScaler.fit(X)
        TargetVarScalerFit=TargetVarScaler.fit(y)

        with open('data/predictorScaler.pkl', 'wb') as file:
            pickle.dump(PredictorScalerFit, file)

        with open('data/targerScaler.pkl', 'wb') as file:
            pickle.dump(TargetVarScalerFit, file)
        
        # # Generating the standardized values of X and y
        X=PredictorScalerFit.transform(X)
        y=TargetVarScalerFit.transform(y)

        # Split the data into training and testing set
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        X_train = X.sample(frac=0.8, random_state=25)
        X_test = X.drop(X_train.index).values
        X_train = X_train.values

        y_train = y.sample(frac=0.8, random_state=25)
        y_test = y.drop(y_train.index).values
        y_train.values
        
        model = Sequential()
        # Defining the first layer of the model
        model.add(Dense(units=6, input_shape=(6,), kernel_initializer='normal', activation='relu'))

        # Defining the Second layer of the model
        model.add(Dense(units=60, kernel_initializer='normal', activation='relu'))
        model.add(Dense(units=60, kernel_initializer='normal', activation='relu'))
        model.add(Dense(units=6, kernel_initializer='normal', activation='relu'))


        # The output neuron is a single fully connected node 
        # Since we will be predicting a single number
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        # Compiling the model
        optimizer = Adam(learning_rate=0.01)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        # Fitting the ANN to the Training set
        model.fit(X_train, y_train ,batch_size = 64, epochs = 200, verbose=0)
        
        model.save("data/model.h5")

        # Generating Predictions on testing data
        Predictions=model.predict(X_test)
        
        # Scaling the predicted Price data back to original price scale
        Predictions=TargetVarScalerFit.inverse_transform(Predictions)
        
        # Scaling the y_test Price data back to original price scale
        y_test_orig=TargetVarScalerFit.inverse_transform(y_test)
        
        # Scaling the test data back to original scale
        Test_Data=PredictorScalerFit.inverse_transform(X_test)
        
        TestingData=pd.DataFrame(data=Test_Data, columns=['Diagnosis', 'Tindakan', 'SEX', 'UMUR_TAHUN', 'Severity', 'HARI'])
        TestingData['LAMA_DIRAWAT']=y_test_orig
        TestingData['PRED']=Predictions
        TestingData.head()
        
        # Computing the absolute percent error
        APE=100*(abs(TestingData['LAMA_DIRAWAT']-TestingData['PRED'])/TestingData['LAMA_DIRAWAT'])
        TestingData['APE']=APE

        ape_result = 100-np.mean(APE)

        return ape_result, model.evaluate(X_test, y_test, verbose=0)
