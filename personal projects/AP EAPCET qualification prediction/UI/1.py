import numpy as np  # If numpy is not imported already

# Assuming 'model' is your trained machine learning model
input_data = np.array([[487, 6, 1]])  # Creating an array with the input values
prediction = model.predict(input_data)  # Making the prediction
output = prediction[0]  # Retrieving the output value

print(output)  # Displaying the output
