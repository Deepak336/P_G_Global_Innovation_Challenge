# Multiple Outputs
import pandas as pd
from keras.models import model_from_json
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense


# load the dataset
datasetX =  pd.read_csv("Datasets/customerTrainingDataset.csv", header=None ,sep=',' , usecols =[0,1,2,3])
datasetY1 = pd.read_csv("Datasets/customerTrainingDataset.csv", header=None ,sep=',' , usecols =[4])
datasetY2 = pd.read_csv("Datasets/customerTrainingDataset.csv", header=None ,sep=',' , usecols =[5])
datasetY3 = pd.read_csv("Datasets/customerTrainingDataset.csv", header=None ,sep=',' , usecols =[6])


# input layer
visible = Input(shape=(4,))

# Hidden layers
#nodes
hidden1 = Dense(12, activation='relu')(visible)
hidden2 = Dense(8, activation='relu')(hidden1)

# classification output 3
cashBack = Dense(1, activation='tanh')(hidden2)
economicalStatus = Dense(1, activation='tanh')(hidden2)
valueCustomer = Dense(1, activation='tanh')(hidden2)

# output
model = Model(inputs=visible, outputs=[cashBack, economicalStatus,valueCustomer])

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

model.fit(datasetX, [datasetY1,datasetY2,datasetY3] , epochs=150, batch_size=10, verbose=0)


# summarize layers
print(model.summary())

# serialize model to JSON
model_json = model.to_json()
with open("customerCashbackModel/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("customerCashbackModel/model.h5")
print("Saved model to disk")
