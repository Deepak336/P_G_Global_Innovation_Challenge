import numpy
import pandas as pd
from keras.models import model_from_json
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense


# load the dataset
datasetX =  pd.read_csv("Datasets/customerPredictDataset.csv", header=None ,sep=',' , usecols =[0,1,2,3])
datasetY1 = pd.read_csv("Datasets/customerPredictDataset.csv", header=None ,sep=',' , usecols =[4])
datasetY2 = pd.read_csv("Datasets/customerPredictDataset.csv", header=None ,sep=',' , usecols =[5])
datasetY3 = pd.read_csv("Datasets/customerPredictDataset.csv", header=None ,sep=',' , usecols =[6])


# load json and create model
json_file = open('customerCashbackModel/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("customerCashbackModel/model.h5")
print("Loaded model from disk")


# evaluate loaded model on test data
loaded_model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
# make class predictions with the model
predictions = loaded_model.predict(datasetX)

# summarize the first 5 cases
for i in range(5):
	#print(datasetX.iloc[i], predictions[i], datasetY1.iloc[i],datasetY2.iloc[i],datasetY3.iloc[i])
    print(predictions[0][i].round(2), predictions[1][i].round(2), predictions[2][i].round(2))
