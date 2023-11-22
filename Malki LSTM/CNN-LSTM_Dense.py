import os
import natsort
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM, Dense, LeakyReLU, ReLU, Dropout, PReLU
from keras.backend import clear_session
from keras.models import Model
from keras.optimizers import Adam, Nadam

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Dropout, Dense, BatchNormalization
from keras.layers import Conv1D, Flatten, MaxPooling1D


# Load your time series data into a pandas DataFrame


directory = r'C:\Users\377022\Desktop\ROMs_WBL\New_Set'
data = pd.read_csv( os.path.join(directory,"OIPM_Case.csv"))

OIPM = np.array(data.pop('OIPM_1')).reshape(-1,1)
scaler_OIPM = MinMaxScaler(feature_range=(0,1))
scaler_OIPM.fit(OIPM)
OIPM_norm = scaler_OIPM.transform(OIPM)
OIPM_norm = OIPM_norm.reshape(-1, 101)




scaler_data = MinMaxScaler(feature_range=(0,1))
scaler_data.fit(data)
data_norm = scaler_data.transform(data)


# Calculate the number of sequences
num_sequences = data_norm.shape[0] // 101


data_norm = data_norm.reshape(-1, 101, 9)





## classifier


# Creating binary label for zero/non-zero series
labels = np.any(OIPM_norm != 0, axis=1)
labels = labels.astype(int)  # Convert boolean to int

# Split your data for classification task
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(data_norm, labels, test_size=0.2)

# Defining CNN model for classification
def make_cnn_model():
    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(101, 9)))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    return model

# Compile and train the CNN
cnn_model = make_cnn_model()
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(X_train_clf, y_train_clf, 
                    epochs=100,
                    batch_size=128, 
                    validation_split=0.2, 
                    shuffle=True, 
                    verbose=True)


# Predicting labels for test data
preds = (cnn_model.predict(X_test_clf) > 0.5).astype("int32")


# Convert predictions to numpy array if they're not already
preds = np.array(preds)

# Create a mask for each class
mask_class_0 = preds == 0
mask_class_1 = preds == 1

# Use the masks to extract the corresponding realizations
P_0 = preds[mask_class_0]
P_1 = preds[mask_class_1]

# Print shapes to verify
print(f"P_0 shape: {P_0.shape}")
print(f"P_1 shape: {P_1.shape}")





# Creating binary label for zero/non-zero series for the entire dataset
labels_all = np.any(OIPM_norm != 0, axis=1)
labels_all = labels_all.astype(int)  # Convert boolean to int

# Fit the CNN classifier on the entire dataset using the binary labels
history = cnn_model.fit(data_norm, labels_all, epochs=100, batch_size=128, shuffle=True, verbose=True)

# Predict the class labels for the entire dataset
preds_all = (cnn_model.predict(data_norm) > 0.5).astype("int32")

# preds_all contain the predicted class labels for your entire dataset

# extract the realizations for class 0 and class 1:
mask_class_0_all = preds_all == 0
mask_class_1_all = preds_all == 1

# Use the masks to extract the corresponding realizations
P_0_all = data_norm[mask_class_0_all.squeeze()]
P_1_all = data_norm[mask_class_1_all.squeeze()]

# Print shapes to verify
print(f"P_0_all shape: {P_0_all.shape}")
print(f"P_1_all shape: {P_1_all.shape}")



# Calculate the mean across features for each realization and timestep
P_1_all_mean = P_1_all.mean(axis=-1)

# Plot each realization
plt.figure(figsize=(10,6))
for i in range(P_1_all_mean.shape[0]):
    plt.plot(P_1_all_mean[i, :], label=f'Realization {i+1}' if i < 10 else None)  # add label for the first 10 realizations

plt.xlabel('Timestep')
plt.ylabel('Mean value of features')
plt.title('P_1_all values for all realizations over time')
plt.legend()
plt.show()




# Predict the class labels for the entire dataset
preds_all = (cnn_model.predict(data_norm) > 0.5).astype("int32")

# If you want to extract the realizations for class 0 and class 1:
mask_class_0_all = preds_all == 0
mask_class_1_all = preds_all == 1

# Use the masks to extract the corresponding realizations
OIPM_norm_1 = OIPM_norm[mask_class_1_all.squeeze()]
data_norm_1 = data_norm[mask_class_1_all.squeeze()]

print(f"data_norm_1 shape: {data_norm_1.shape}")
print(f"OIPM_norm_1 shape: {OIPM_norm_1.shape}")


realization_index = 60  # Select the 11th realization (0-indexed)

plt.figure(figsize=(10,6))
plt.plot(OIPM_norm_1[realization_index, :], label=f'Realization {realization_index+1}')
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.title(f'Temporal Evolution of Realization {realization_index+1}')
plt.legend()
plt.show()



plt.figure(figsize=(10,6))

for i in range(OIPM_norm_1.shape[0]):
    plt.plot(OIPM_norm_1[i, :], label=f'Realization {i+1}')

plt.xlabel('Timestep')
plt.ylabel('Value')
plt.title('Temporal Evolution of All Realizations')
#plt.ylim([0, 0.01])
#plt.xlim([99, 101])
plt.show()


# scaling back the data
data_1 = data_norm_1.reshape(-1, data_norm_1.shape[-1])
data_1 = scaler_data.inverse_transform(data_1)
data_1 = data_1.reshape(data_norm_1.shape)

# scaling back the output
OIPM_1 = OIPM_norm_1.reshape(-1, 1)
OIPM_1 = scaler_OIPM.inverse_transform(OIPM_1)
OIPM_1 = OIPM_1.reshape(OIPM_norm_1.shape)



plt.figure(figsize=(10,6))
for i in range(OIPM_1.shape[0]):
    plt.plot(OIPM_1[i, :])
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.title('Temporal Evolution of All Realizations unsclaed')
#plt.ylim([0, 10000])
#plt.xlim([99, 100])
plt.show()







############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
#%%
# Set the seed for tensorflow random numbers
tf.random.set_seed(242)


X_train, X_test, y_train, y_test = train_test_split(data_norm_1, OIPM_norm_1, test_size=0.2)

from keras.regularizers import L2
from tensorflow.keras.activations import swish
import tensorflow_addons as tfa
from tensorflow.keras.layers import PReLU
    
def lstm_OIPM_model():
    clear_session()
    inp = Input(shape=(101,9))
    _ = LSTM(9, kernel_regularizer=L2(1e-5), dropout=0.1, return_sequences=True)(inp)
    _ = PReLU()(_)
    _ = BatchNormalization()(_)
    #_ = Dropout(0.4)(_)
    #, batch_first=True
    
    _ = LSTM(16, kernel_regularizer=L2(1e-5), dropout=0.1, return_sequences=True)(_)
    _ = PReLU()(_)
    _ = BatchNormalization()(_)
    
    _ = LSTM(32, kernel_regularizer=L2(1e-5), dropout=0.1, return_sequences=True)(_)
    _ = PReLU()(_)
    _ = BatchNormalization()(_)
    
    _ = LSTM(64, kernel_regularizer=L2(1e-5), dropout=0.1, return_sequences=False)(_)
    _ = PReLU()(_)
    _ = BatchNormalization()(_)
    
    '''
    _ = LSTM(8, kernel_regularizer=L2(0.001), dropout=0.1)(_)
    #, return_sequences=True
    #kernel_regularizer=L2(0.001)
    '''
    '''
    _ = Dense(128, kernel_regularizer=L2(0.001))(_)
    _ = Dropout(0.5)(_)
    _ = ReLU()(_)
    _ = BatchNormalization()(_)
    
    
    _ = Dense(64, kernel_regularizer=L2(0.001))(_)
    _ = Dropout(0.4)(_)
    _ = ReLU()(_)
    #_ = PReLU()(_)
    #_ = swish(_)
    #_ = tfa.activations.gelu(_)
    _ = BatchNormalization()(_)
    '''
      
    


    _ = Dense(128, kernel_regularizer=L2(1e-5))(_)
    _ = Dropout(0.1)(_)
    #_ = ReLU()(_)
    _ = PReLU()(_)
    #_ = swish(_)
    #_ = tfa.activations.gelu(_)
    _ = BatchNormalization()(_)
    
    #_ = BatchNormalization(momentum=0.1, epsilon=1e-05, center=True, scale=True)(_)
    
    
    
    
   
    _ = Dense(64, kernel_regularizer=L2(1e-5))(_)
    _ = Dropout(0.1)(_)
    #_ = ReLU()(_)
    _ = PReLU()(_)
    #_ = swish(_)
    #_ = tfa.activations.gelu(_)
    _ = BatchNormalization()(_)
    
    _ = Dense(32, kernel_regularizer=L2(1e-5))(_)
    _ = Dropout(0.1)(_)
    #_ = ReLU()(_)
    _ = PReLU()(_)
    #_ = swish(_)
    #_ = tfa.activations.gelu(_)
    _ = BatchNormalization()(_)
    
    _ = Dense(101)(_)
    out = _
    return Model(inp, out)




model = lstm_OIPM_model()


model.compile(optimizer=Nadam(learning_rate=1e-3), loss='mse', metrics=['mse','mae'])

#beta_1=0.9, beta_2=0.999
#learning_rate=1e-4, decay=1e-5


history = model.fit(X_train, y_train, 
                    epochs=300, 
                    batch_size=100, 
                    validation_split=0.2, 
                    shuffle=True, 
                    verbose=True)

y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)


mse = mean_squared_error(y_train, y_train_pred)
print('MSE: {:.3f}'.format(mse))




# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0,0.06)
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# summarize history for mse
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# summarize history for mae
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()







#############
plt.figure()
plt.plot(np.mean(y_test, axis=0), np.mean(y_test_pred, axis=0), 'k--')
plt.xlim([0,0.2]); plt.ylim([0,0.2])
plt.title('Mean of Predictions vs. Actual')
plt.show()

tstep = 100
plt.figure()
plt.plot(y_test[:,tstep], y_test_pred[:,tstep], '.')
plt.xlabel('True'); plt.ylabel('Pred')
plt.xlim([0,1]); plt.ylim([0,1])
plt.title(f'Predictions vs. Actual at Timestep {tstep}')
plt.show()


k = 20
timesteps = np.linspace(0, 1000, num=101)
plt.figure()
plt.plot(timesteps, y_test[k,:], label='true')
plt.plot(timesteps, y_test_pred[k,:], label='pred')
plt.xlabel('time [years]'); plt.ylabel('OIPM')
plt.legend()
plt.title(f'Comparison between Predictions vs. Actual for case {k+1}')
plt.show()




k = 54
timesteps = np.linspace(0, 1000, num=101)
plt.figure()
plt.plot(timesteps, y_train[k,:], label='true')
plt.plot(timesteps, y_train_pred[k,:], label='pred')
plt.xlabel('time [years]'); plt.ylabel('OIPM')
plt.legend()
plt.show()



plt.figure(figsize=(10, 6))
for i in range(y_test.shape[0]):
    plt.plot(y_test_pred[i, :], label=f'Realization {i+1}')

plt.xlabel('Timestep')
plt.ylabel('Prediction')
plt.title('Predictions of All Realizations')
plt.show()






import random
timesteps = np.linspace(0, 1000, num=101)

# Loop for 10 random plots
for i in range(10):
    k = random.randint(0, 100)  # Randomly select a case between 0 and 100

    plt.figure()
    plt.plot(timesteps, y_test[k,:], label='true')
    plt.plot(timesteps, y_test_pred[k,:], label='pred')
    plt.xlabel('time [years]'); plt.ylabel('OIPM')
    plt.legend()
    plt.title(f'Comparison between Predictions vs. Actual for case {k+1}')
    plt.show()






'''
plt.figure(figsize=(10, 6))

for i in range(y_test.shape[0]):
    plt.plot(y_test_pred[i, :], label=f'Realization {i+1}')

plt.xlabel('Timestep')
plt.ylabel('Prediction')
plt.ylim([0, 0.01])
plt.title('Predictions of All Realizations')
plt.show()
'''


timesteps = np.linspace(0, 1000, num=101)
plt.figure(figsize=(10, 6))
for i in range(y_test.shape[0]):
    plt.plot(y_test[i, :], y_test_pred[i, :], '.')
plt.plot([0, 1], [0, 1], 'k--',  linewidth=4)
plt.xlabel('True')
plt.ylabel('Prediction')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
plt.title('True vs Prediction for All Realizations')
plt.show()



timesteps = np.linspace(0, 1000, num=101)
plt.figure(figsize=(10, 6))
for i in range(y_test.shape[0]):
    plt.plot(y_test[i, :], y_test_pred[i, :], '.')
plt.plot([0, 1], [0, 1], 'k--',  linewidth=4)
plt.xlabel('True')
plt.ylabel('Prediction')
plt.xlim([0, 0.2])
plt.ylim([0, 0.2])
plt.title('True vs Prediction for All Realizations')
plt.show()












plt.figure(figsize=(10, 6))
for i in range(y_train.shape[0]):
    plt.plot(y_train[i, :], y_train_pred[i, :], '.')

plt.plot([0, 1], [0, 1], 'k--',  linewidth=4)
plt.xlabel('True')
plt.ylabel('Prediction')
#plt.xlim([0, 0.2])
#plt.ylim([0, 0.2])
plt.title('True vs Prediction using training data')
plt.show()

    
    
'''
plt.figure(figsize=(10, 6))
for i in range(y_train.shape[0]):
    #plt.plot(timesteps, y_train[i, :], label='True')
    plt.plot(timesteps, y_train_pred[i, :], label='Prediction')
plt.xlabel('Time steps')
plt.ylabel('Values')
plt.title('True vs Prediction for All Realizations for training')
plt.show()
'''









timesteps = np.linspace(0, 1000, num=101)

plt.figure(figsize=(10, 6))

for i in range(y_train.shape[0]):
    plt.plot(y_train[i, :], y_train_pred[i, :], '.')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('True')
plt.ylabel('Prediction')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('True vs Prediction for All Realizations for training')
plt.show()


















timesteps = np.linspace(0, 1000, num=101)

plt.figure(figsize=(10, 6))

for i in range(y_train.shape[0]):
    plt.plot(y_train[i, :], y_train_pred[i, :], '.')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('True')
plt.ylabel('Prediction')
plt.xlim([0, 0.2])
plt.ylim([0, 0.2])
plt.title('True vs Prediction for All Realizations for training')
plt.show()




################################################################################################################################
################################################################################################################################
################################################################################################################################


# scaling back the predictions
y_test_pred_reshaped = y_test_pred.reshape(-1, 1)
y_test_pred_original = scaler_OIPM.inverse_transform(y_test_pred_reshaped)
y_test_pred_original = y_test_pred_original.reshape(y_test_pred.shape)











'''
# Assuming y_test_pred is your normalized predictions array and has the same number of samples as your OIPM_norm
# First, you need to reshape it to (-1, 1) because the scaler expects a 2D array
y_test_pred_reshaped = y_test_pred.reshape(-1, 1)

# Now, use the inverse_transform method of the scaler to scale back to original values
y_test_pred_original = scaler_OIPM.inverse_transform(y_test_pred_reshaped)

# If the original predictions were in the shape of OIPM_norm, you would reshape it back to that shape
y_test_pred_original = y_test_pred_original.reshape(OIPM_norm.shape)


# Assuming y_test_pred is your predictions array and has the same number of rows as OIPM_norm_1
y_test_pred_reshaped = y_test_pred.reshape(-1, 1)
y_test_pred_original = scaler_OIPM.inverse_transform(y_test_pred_reshaped)
y_test_pred_original = y_test_pred_original.reshape(y_test_pred.shape)
'''






'''
scaler_OIPM = MinMaxScaler(feature_range=(0,1))

# scaling back the data
data_1 = data_norm_1.reshape(-1, data_norm_1.shape[-1])
data_1 = scaler_data.inverse_transform(data_1)
data_1 = data_1.reshape(data_norm_1.shape)

# scaling back the output
OIPM_1 = OIPM_norm_1.reshape(-1, 1)
OIPM_1 = scaler_OIPM.inverse_transform(OIPM_1)
OIPM_1 = OIPM_1.reshape(OIPM_norm_1.shape)
'''




















# Plot for all realizations
plt.figure(figsize=(10, 6))

# Loop through all realizations
for i in range(y_test.shape[0]):
    plt.plot(y_test[i, :], y_test_pred_original[i, :], '.', label=f'Realization {i+1}')

# Identity line for reference
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)

plt.xlabel('True OIPM')
plt.ylabel('Predicted OIPM')
plt.title('True vs Predicted OIPM for All Realizations on Test Data')
plt.show()


# Specify the realization case you want to plot
k = 20  # Change this to the case number you're interested in

plt.figure(figsize=(10, 6))

# Plot the specific case
plt.plot(y_test[k, :], y_test_pred_original[k, :], 'o', label=f'Case {k+1}')

# Identity line for reference
plt.plot([y_test[k, :].min(), y_test[k, :].max()], [y_test[k, :].min(), y_test[k, :].max()], 'k--', linewidth=2)

plt.xlabel('True OIPM')
plt.ylabel('Predicted OIPM')
plt.title(f'True vs Predicted OIPM for Case {k+1}')
plt.legend()
plt.show()






