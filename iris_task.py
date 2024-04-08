
import numpy as np
import matplotlib.pyplot as plt

# Sepal Length, Sepal Width, Petal Length, Petal Width

def read_file(txt_file):
    with open(txt_file, 'r') as file:

        data = file.readlines()

        iris_data = []

        #Append each line from data in the iris_data to create an array
        for line in data:
                iris_data.append([float(num) for num in line.split(',')])

        #Splitting into training and test data
        training_data = iris_data[:30]
        test_data = iris_data[30:]
  
    return training_data, test_data

class_1_training, class_1_test = read_file('class_1')
class_2_training, class_2_test = read_file('class_2')
class_3_training, class_3_test = read_file('class_3')

#The 
def sigmoid(z):
     return 1/(1+np.e(-z))