import numpy as np
import pickle 

# Loading the saved model
loaded_model = pickle.load(open('D:/Python Projects/Diabetes Prediction System/Google Colab/trained_model.sav','rb'))

input_data = (1,85,66,29,0,26.6,0.351,31)

# Changing the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# Reshaping the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The person is not Diabetic')
else:
  print('The person is Diabetic')