# General data analysis/plotting
import numpy as np
import pandas as pd
import pickle
import sqlite3
from sklearn.model_selection import train_test_split

# Data preprocessing
from sklearn.preprocessing import StandardScaler

# Neural Net modules
from keras.models import Sequential
from keras.layers import Dense

class NeuralNetwork:
    def createModel(self):
        conn = sqlite3.connect("data/rsdb.db")
        df = pd.read_sql_query("SELECT * from rs_table", conn)
        conn.close()

        df = df[['Diagnosis', 'Tindakan', 'SEX', 'UMUR_TAHUN', 'LAMA_DIRAWAT']]

        df['DiagnosisCAT'] = df['Diagnosis']
        df['Diagnosis'] = df['DiagnosisCAT'].astype('category')
        df['Diagnosis'] = df['Diagnosis'].cat.reorder_categories(df['DiagnosisCAT'].unique(), ordered=True)
        df['Diagnosis'] = df['Diagnosis'].cat.codes

        df['TindakanCAT'] = df['Tindakan']
        df['Tindakan'] = df['TindakanCAT'].astype('category')
        df['Tindakan'] = df['Tindakan'].cat.reorder_categories(df['TindakanCAT'].unique(), ordered=True)
        df['Tindakan'] = df['Tindakan'].cat.codes

        df.dropna(axis=0, inplace=True)

        X = df[['Diagnosis', 'Tindakan', 'SEX', 'UMUR_TAHUN']].values
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = Sequential()
        # Defining the first layer of the model
        model.add(Dense(units=4, input_shape=(4,), kernel_initializer='normal', activation='relu'))

        # Defining the Second layer of the model
        model.add(Dense(units=60, kernel_initializer='normal', activation='relu'))
        model.add(Dense(units=60, kernel_initializer='normal', activation='relu'))
        model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))


        # The output neuron is a single fully connected node 
        # Since we will be predicting a single number
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        # Compiling the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Fitting the ANN to the Training set
        model.fit(X_train, y_train ,batch_size = 200, epochs = 500, verbose=0)
        
        model.save("data/model.h5")

        # Generating Predictions on testing data
        Predictions=model.predict(X_test)
        
        # Scaling the predicted Price data back to original price scale
        Predictions=TargetVarScalerFit.inverse_transform(Predictions)
        
        # Scaling the y_test Price data back to original price scale
        y_test_orig=TargetVarScalerFit.inverse_transform(y_test)
        
        # Scaling the test data back to original scale
        Test_Data=PredictorScalerFit.inverse_transform(X_test)
        
        TestingData=pd.DataFrame(data=Test_Data, columns=['Diagnosis', 'Tindakan', 'SEX', 'UMUR_TAHUN'])
        TestingData['LAMA_DIRAWAT']=y_test_orig
        TestingData['PRED']=Predictions
        TestingData.head()
        
        # Computing the absolute percent error
        APE=100*(abs(TestingData['LAMA_DIRAWAT']-TestingData['PRED'])/TestingData['LAMA_DIRAWAT'])
        TestingData['APE']=APE

        ape_result = 100-np.mean(APE)

        return ape_result, model.evaluate(X_test, y_test, verbose=0)
