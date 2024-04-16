
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, accuracy_score

#The vowels are fetched from vowdata_nohead.dat
#We only need column 4, 5, and 6, which is frequency f1, f2 and f3. They are "steady state"

# ae, ah, aw, eh, er, ei, ih, iy, oa, oo, uh, uw

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


def mean(training_data):
    mu_1 = np.mean(training_data[:, 0])
    mu_2 = np.mean(training_data[:, 1])
    mu_3 = np.mean(training_data[:, 2])
    mu = [mu_1, mu_2, mu_3]
    return mu

def cov_matrix(training_data):
     return np.cov(training_data, rowvar=False)/len(training_data[:, 0])

def multivariate_gaussian(x, mean, cov):
    return multivariate_normal(mean=mean, cov=cov).pdf(x)

print("ae training set: ", ae_training)
print("Mean value vector: ", mean(ae_training))


