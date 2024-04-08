
import numpy as np
import matplotlib.pyplot as plt

# Sepal Length, Sepal Width, Petal Length, Petal Width

def read_file(txt_file):
    with open(txt_file, 'r') as file:

        content = file.read()

        content_list = content.split()

        content_list = [float(value) for value in content_list]
  
    return np.array(content_list)
