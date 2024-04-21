
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Sepal Length, Sepal Width, Petal Length, Petal Width

def read_file(txt_file):
    with open(txt_file, 'r') as file:

        data = file.readlines()

        iris_data = []

        #Append each line from data in the iris_data to create an array
        for line in data:
                iris_data.append([float(num) for num in line.split(',')])

        for row in iris_data:
             row.append(1)

        #Splitting into training and test data
        training_data = iris_data[:30]
        test_data = iris_data[30:]
  
    return np.array(iris_data), np.array(training_data), np.array(test_data)

class_1, class_1_training, class_1_test = read_file('Iris_TTT4275/Iris_TTT4275/class_1')
class_2, class_2_training, class_2_test = read_file('Iris_TTT4275/Iris_TTT4275/class_2')
class_3, class_3_training, class_3_test = read_file('Iris_TTT4275/Iris_TTT4275/class_3')

# Concatenate training and test datasets
training_data = np.concatenate((class_1_training, class_2_training, class_3_training), axis=0)
test_data = np.concatenate((class_1_test, class_2_test, class_3_test), axis=0)

def sigmoid(z):
     return (np.exp(z))/(1+np.exp(z))

def g_k(W, x_k):
     z = np.dot(W, x_k)
     return sigmoid(z)

#Initializing the weigth matrix. Each element is initially random between 0 and 0.01
W = np.random.rand(3, 5)*0.01

#g_k er estimert verdi fra overnevnte funksjon
#t_k er target vector. Så om input x_k er av class 1, vil t_k=[1, 0, 0]
#x_k er input vektoren
def grad_MSE(W, training_data):
     
     #Initial values for MSE and its gradient matrix
     grad_MSE_matrix = np.zeros((3, 5))
     MSE = 0

     #Checks which class the target output is
     for k in range(len(training_data)):
          if k < 30:
               t = np.array([1, 0, 0])
          elif 30 <= k < 60:
               t = np.array([0, 1, 0])
          else:
               t = np.array([0, 0, 1])

          #Computing values for g, x_k and column vector used in outer product
          g = g_k(W, training_data[k])
          col_vector = ((g - t) * g * (1 - g))
          x_k = np.array(training_data[k])

          # Computing the MSE and its gradient
          MSE += 0.5 * np.inner(g-t, g-t)
          grad_MSE_matrix += np.outer(col_vector, x_k)

     return grad_MSE_matrix, MSE


def training(W, training_data):
     
     #Initial values used in training
     alpha = 0.03
     MSE = grad_MSE(W, training_data)[1]
     iteration = 0

     #Training until the MSE reaches disired threshold
     while MSE > 9:

          #Updating the W matrix
          W -= alpha * grad_MSE(W, training_data)[0]

          #Updating the MSE     
          MSE = grad_MSE(W, training_data)[1]

          iteration += 1

     print("Number of iterations: ", iteration)
     print("Final MSE:", MSE)

     return W

def test(W, test_data):
     
     pred = []
     
     for x_k in test_data:
          g = g_k(W, x_k)
          
          #+1 since python is zero indexed
          pred.append(np.argmax(g)+1)

     return pred


def conf_matrix(test_data, pred):

     true = []

     for k in range(len(test_data)):
          if k < 20:
               true.append(1)
          elif 20 <= k < 40:
               true.append(2)
          else:
               true.append(3)

     cm = confusion_matrix(true, pred)
     error = 1 - accuracy_score(true, pred)

     return cm, error


#                                              Training phase
W_trained = training(W, training_data)
print("Finally trained W matrix:", W_trained)

#                       Test phase. Returns a 3x1-list with the number of classified samples for each class
pred = test(W_trained, test_data)

#                                           Confussion matrix
m, error = conf_matrix(test_data, pred)
print("Confusion matrix:", m)
print("Error rate:", error)


def plot():
     fig, axs = plt.subplots(4, 4)
     features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
     for i in range(4):
          for j in range(4):
               if i == j:
                    ax = axs[i, j]
                    ax.axis('off')  # Hide axes
                    ax.text(0.5, 0.5, f'{features[i]}', ha='center', va='center', fontsize=12)
                    continue

               ax = axs[i, j]
               ax.scatter(class_1[:, j], class_1[:, i], color='red', label='Iris-setosa', s=10)
               ax.scatter(class_2[:, j], class_2[:, i], color='green', label='Iris-versicolor', s=10)
               ax.scatter(class_3[:, j], class_3[:, i], color='blue', label='Iris-virginica', s=10)

               if i==0 and j==3:
                    ax.legend()
     plt.show()

#plot()

def histogram():
     fig, axs = plt.subplots(4, 1)
     features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
     bin_width = 0.2
     for i in range(4):
          ax = axs[i]
          bins = np.arange(0, 8 + bin_width, bin_width)
          ax.hist(class_1[:, i], bins=bins, color='red', label='Iris-setosa', alpha=0.5)#, range=(0, 8))
          ax.hist(class_2[:, i], bins=bins, color='green', label='Iris-versicolor', alpha=0.5)#, range=(0, 8))
          ax.hist(class_3[:, i], bins=bins, color='blue', label='Iris-virginica', alpha=0.5)#, range=(0, 8))

          ax.set_xlabel(f'{features[i]}')
     plt.show()

#histogram()