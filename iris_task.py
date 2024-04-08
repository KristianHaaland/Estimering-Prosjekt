
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

class_1, class_1_training, class_1_test = read_file('class_1')
class_2, class_2_training, class_2_test = read_file('class_2')
class_3, class_3_training, class_3_test = read_file('class_3')
#training, test = read_file('iris.data')

#z=Wx
def sigmoid(z):
     return 1/(1+np.e(-z))

#Initializing the weigth matrix. Each element is initially random between 0 and 0.01
W = np.random.rand(3, 4) * 0.01


def g_k(W, x):
     return np.dot(W, x)

#g_k er estimert verdi fra overnevnte funksjon
#t_k er target vector. Så om input x_k er av class 1, vil t_k=[1, 0, 0]
#x_k er input vektoren
def grad_MSE(g, t, x):
     
     sum_elements = 0

     for i in range(4):
          sum_elements += (g[i]-t[i])*g[i]*(1-g[i])*x[i]

     return sum_elements

def training():
     
    return 0

#Scatter plots
def plot(training_data):
     
     sepal_lengths = [elem[0] for elem in training_data]
     sepal_widths = [elem[1] for elem in training_data]
     petal_lengths = [elem[2] for elem in training_data]
     petal_widths = [elem[3] for elem in training_data]

     plt.xlabel('Sepal Length')
     plt.ylabel('Sepal Width')
     plt.grid()
     #plt.title()

     return plt.scatter(petal_lengths, petal_widths)

#plot(class_1)
#plot(class_2)
#plot(class_3)
#plt.show()