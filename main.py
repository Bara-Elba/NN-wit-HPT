from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf

from kerastuner.tuners import Hyperband

# from keras.wrappers.scikit_learn import KerasRegressor
#import lightgbm as lgb
# import matplotlib.pyplot as plt 
# import seaborn as sns
# import numpy as np
import pandas as pd
import time
from model import ANNhypermodel 


if __name__ == "__main__":
    
    # read the data from an xlsx file
    data = pd.read_excel('dataset/data.xlsx')
    relevant_columns = ['Yield', 'CanopyArea'] + [col for col in data.columns if 'mean' in col or 'sum' in col]
    filtered_data = data[relevant_columns].dropna()

    # Applying one-hot encoding
    df_encoded = pd.get_dummies(data['Site '], dtype=int)

    #Normalization

    scaler = MinMaxScaler()
    data_normalized = pd.DataFrame(scaler.fit_transform(filtered_data), columns=filtered_data.columns)
    
    result = pd.concat([data_normalized, df_encoded], axis=1)

    X = data_normalized.drop('Yield', axis=1)

    y = result['Yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    hypermodel= ANNhypermodel(input_shape= [len(X_train.keys())])

    HYPERBAND_MAX_EPOCHS = 10
    EXECUTION_PER_TRIAL = 3

    tuner= Hyperband(hypermodel,
                   objective= 'val_mse',
                   max_epochs=HYPERBAND_MAX_EPOCHS, #Set 100+ for good results
                   executions_per_trial=EXECUTION_PER_TRIAL,
                   directory= 'hyperband',
                   project_name='houseprices',
                   overwrite=True)
    
    
    print('searching for the best params!')

    t0= time()
    tuner.search(x= X_train,
                y= y_train,
                epochs=100,
                batch_size= 8,
                validation_data= (X_test, y_test),
                verbose=1,
                callbacks= []
                )
    print(time()- t0," secs")

    # Retreive the optimal hyperparameters
    best_hps= tuner.get_best_hyperparameters(num_trials=1)[0]

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    print(" \n ")
    print(f"""
    The hyperparameter search is complete. The optimal number of units in the
    first densely-connected layer is {best_hps.get('units_1')},
    second layer is {best_hps.get('units_2')}
    third layer is {best_hps.get('units_3')}

    and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    # Evaluate the best model.
    print(best_model.metrics_names)
    loss, mae, mse = best_model.evaluate(X_test, y_test)
    print(f'loss:{loss} mae: {mae} mse: {mse}')
