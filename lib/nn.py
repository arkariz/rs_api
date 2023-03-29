# General data analysis/plotting
import pandas as pd
import pickle
import sqlite3

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
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
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
        model.fit(X, y ,batch_size = 200, epochs = 500, verbose=0)
        
        model.save("data/model.h5")
        print("ok")
