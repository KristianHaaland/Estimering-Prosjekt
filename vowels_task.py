
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, accuracy_score

#The vowels are fetched from vowdata_nohead.dat
#We only need column 4, 5, and 6, which is frequency f1, f2 and f3. They are "steady state"

# ae, ah, aw, eh, er, ei, ih, iy, oa, oo, uh, uw

"""------------------------ DATA FETCHING --------------------------------------------------------------------------------------------------------------"""
def read_file(txt_file):
    with open(txt_file, 'r') as file:

        data = file.readlines()

        vowel_data = []

        #Append each line from data in the iris_data to create an array
        for line in data:
                values = line.split()[1:]

                vowel_data.append([float(num) for num in values])
  
    return np.array(vowel_data)[:, 2:5]

vowel_data = read_file('Wovels/Wovels/vowdata_nohead.dat')  

ae_data = vowel_data[0:139]
ah_data = vowel_data[139:278]
aw_data = vowel_data[278:417]
eh_data = vowel_data[417:556]
er_data = vowel_data[556:695]
ei_data = vowel_data[695:834]
ih_data = vowel_data[834:973]
iy_data = vowel_data[973:1112]
oa_data = vowel_data[1112:1251]
oo_data = vowel_data[1251:1390]
uh_data = vowel_data[1390:1529]
uw_data = vowel_data[1529:1668]

ae_training, ae_test = ae_data[:70], ae_data[70:]
ah_training, ah_test = ah_data[:70], ah_data[70:]
aw_training, aw_test = aw_data[:70], aw_data[70:]
eh_training, eh_test = eh_data[:70], eh_data[70:]
er_training, er_test = er_data[:70], er_data[70:]
ei_training, ei_test = ei_data[:70], ei_data[70:]
ih_training, ih_test = ih_data[:70], ih_data[70:]
iy_training, iy_test = iy_data[:70], iy_data[70:]
oa_training, oa_test = oa_data[:70], oa_data[70:]
oo_training, oo_test = oo_data[:70], oo_data[70:]
uh_training, uh_test = uh_data[:70], uh_data[70:]
uw_training, uw_test = uw_data[:70], uw_data[70:]

test_data = np.concatenate((ae_test, ah_test, aw_test, eh_test, er_test, ei_test, ih_test, iy_test, oa_test, oo_test, uh_test, uw_test), axis=0)
"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""

#A priori probabilities. We have the same amount of test samples (69) for each class
P_w = 69/(12*69)

#Calculating the mean-vector for training_data. The components are the mean values of the main frequencies
#which describes the vowel
def mean(training_data):
    mu_1 = np.mean(training_data[:, 0])
    mu_2 = np.mean(training_data[:, 1])
    mu_3 = np.mean(training_data[:, 2])
    mu = [mu_1, mu_2, mu_3]
    return mu

#Calculating the covariance matrix for training_data
def cov_matrix(training_data):
     cov_matrix = np.cov(training_data, rowvar=False)/len(training_data[:, 0])

     #Task 1c, only using the diagonal terms and removing the covariance terms
     diag_matrix = np.diag(np.diag(cov_matrix))

     return diag_matrix


#Dictionary which holds the estimated mean value and covariance matrix for each vowel
trained_parameters = {
    'ae': {'class': 1,  'mean': mean(ae_training), 'covariance': cov_matrix(ae_training)},
    'ah': {'class': 2,  'mean': mean(ah_training), 'covariance': cov_matrix(ah_training)},
    'aw': {'class': 3,  'mean': mean(aw_training), 'covariance': cov_matrix(aw_training)},
    'eh': {'class': 4,  'mean': mean(eh_training), 'covariance': cov_matrix(eh_training)},
    'er': {'class': 5,  'mean': mean(er_training), 'covariance': cov_matrix(er_training)},
    'ei': {'class': 6,  'mean': mean(ei_training), 'covariance': cov_matrix(ei_training)},
    'ih': {'class': 7,  'mean': mean(ih_training), 'covariance': cov_matrix(ih_training)},
    'iy': {'class': 8,  'mean': mean(iy_training), 'covariance': cov_matrix(iy_training)},
    'oa': {'class': 9,  'mean': mean(oa_training), 'covariance': cov_matrix(oa_training)},
    'oo': {'class': 10, 'mean': mean(oo_training), 'covariance': cov_matrix(oo_training)},
    'uh': {'class': 11, 'mean': mean(uh_training), 'covariance': cov_matrix(uh_training)},
    'uw': {'class': 12, 'mean': mean(uw_training), 'covariance': cov_matrix(uw_training)}
}

def test(test_data):

    #Initializing lists and dictionary which stores the results of the test
    predicted_labels = []
    predicted_class = []
    classification_counts = {vowel: 0 for vowel in trained_parameters}

    #Looping through all the test data
    for i in range(len(test_data)):

        max_prob = 0

        #Looping through the different gaussians. Vowels are the keys, parameters are values
        for vowel, parameters in trained_parameters.items():
            mean = parameters['mean']
            cov = parameters['covariance']
            prob = multivariate_normal(mean=mean, cov=cov).pdf(test_data[i])

            #GMM should maybe be here

            #Descision rule. The gauss which gives the largest probability is the class which is chosen
            if prob*P_w > max_prob:
                max_prob = prob*P_w
                class_num = parameters['class']
                predicted_vowel = vowel

        #Appending result to lists and dictionary
        predicted_labels.append(predicted_vowel)
        predicted_class.append(class_num)
        classification_counts[predicted_vowel] += 1

    return predicted_labels, predicted_class, classification_counts

def conf_matrix(pred):
     true = [i for i in range(1, 13) for _ in range(69)]

     cm = confusion_matrix(true, pred)
     error = accuracy_score(true, pred)

     return cm, error

print(test(test_data)[0], '\n')
print(test(test_data)[1], '\n')
print(test(test_data)[2], '\n')

predicted = test(test_data)[1]
print("Confusion matrix: \n", conf_matrix(predicted)[0])

#print("ae training set: ", ae_training)
#print("Mean value vector: ", mean(ae_training))
#print("Covariance matrix: ", cov_matrix(ae_training))
#print(multivariate_gaussian(mean(ae_training), cov_matrix(ae_training)))
