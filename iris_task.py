
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
  
    return iris_data, training_data, test_data

class_1, class_1_training, class_1_test = read_file('class_1')
class_2, class_2_training, class_2_test = read_file('class_2')
class_3, class_3_training, class_3_test = read_file('class_3')
training_data = np.concatenate((class_1_training, class_2_training, class_3_training), axis=0)
test_data = np.concatenate((class_1_test, class_2_test, class_3_test), axis=0)

def sigmoid(z):
     return 1/(1+np.exp(-z))

def g_k(W, x_k):
     z = np.dot(W, x_k)
     return sigmoid(z)

#Initializing the weigth matrix. Each element is initially random between 0 and 0.01
W = np.random.rand(3, 4) * 0.01

#g_k er estimert verdi fra overnevnte funksjon
#t_k er target vector. Så om input x_k er av class 1, vil t_k=[1, 0, 0]
#x_k er input vektoren
def grad_MSE(W, training_data):
     
     #Initial values for MSE and its gradient matrix
     grad_MSE_matrix = np.zeros((3, 4))
     MSE = 0

     #Checks which class the target output is
     for k in range(len(training_data)-1):
          if k < 30:
               t = np.array([1, 0, 0])
          elif 30 <= k < 60:
               t = np.array([0, 1, 0])
          else:
               t = np.array([0, 0, 1])

          #Computing values for g, x_k and column vector used in outer product
          g = g_k(W, training_data[k])
          x_k = training_data[k].reshape(1, -1)
          col_vector = ((g - t) * g * (1 - g)).reshape(-1, 1)

          #Computing the MSE and its gradient
          MSE += 0.5 * np.dot((g-t).T, g-t)
          grad_MSE_matrix += np.dot(col_vector, x_k)

     return grad_MSE_matrix, MSE


def training(W, training_data):
     
     #Initial values used in training
     alpha = .002
     MSE = grad_MSE(W, training_data)[1]
     iteration = 0

     #Training until the MSE reaches disired threshold
     while MSE > 9:

          #Iterating the training data for current W matrix
          for i in range(len(training_data)):
               #Updating the W matrix
               W -= alpha * grad_MSE(W, training_data)[0]

          #Updating the MSE     
          MSE = grad_MSE(W, training_data)[1]
          print(MSE)

          iteration += 1
     #print(iteration)

     return W

#print(training(W, training_data))

def test(W, test_data):
         
         for x_k in test_data:
              g = g_k(W, x_k)
              print(np.argmax(g))

#W_trained = training(W, training_data)
#test(W_trained, test_data)




#Til neste gang:

#Sjekke om det er klassifisert riktig eller ikke
#Lage confussion matrix
#Finne error rate
#Bytte mengde for trening og test


#Scatter plots
# def plot(training_data, legend_label):
     
#      sepal_lengths = [elem[0] for elem in training_data]
#      sepal_widths = [elem[1] for elem in training_data]
#      petal_lengths = [elem[2] for elem in training_data]
#      petal_widths = [elem[3] for elem in training_data]

#      plt.xlabel('Sepal Length')
#      plt.ylabel('Sepal Width')
#      plt.grid()
#      #plt.title()

#      return plt.scatter(petal_lengths, petal_widths, label = legend_label)

# plot(class_1, 'Iris-setosa')
# plot(class_2, 'Iris-vercicolor')
# plot(class_3, 'Iris-virginica')
# plt.legend()
# plt.show()

