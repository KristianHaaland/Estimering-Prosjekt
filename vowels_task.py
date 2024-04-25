
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D

#The vowels are fetched from vowdata_nohead.dat
#We only need column 4, 5, and 6, which is frequency f1, f2 and f3. They are "steady state"

# ae, ah, aw, eh, er, ei, ih, iy, oa, oo, uh, uw

"""---------------------- DATA FETCHING --------------------------------"""
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
"""---------------------------------------------------------------"""

"""------------------------ CONSTANTS ----------------------------"""
P_w = 1/12 #A priori probabilities. We have the same amount of test samples (69) for each class (69/(12*69))
num_gaussians = 3 #How many gaussians we should fit in the GMM
cov_type = "diag" #Covariance matrix used in the GMM
seed = 42 #The GaussianMixture is random each time. By setting a seed we ensure reproducability  
"""---------------------------------------------------------------"""


"""------------------- TRAINING FUNCTIONS ------------------------"""
#Calculating the mean-vector for training_data. The components are the mean values of the formants
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

def GMM(num_gaussians, cov_type, seed, data):
     return GaussianMixture(n_components=num_gaussians, covariance_type = cov_type, random_state=seed).fit(data)
"""---------------------------------------------------------------"""


#Dictionary which holds the estimated mean value and covariance matrix for each vowel
trained_parameters = {
    'ae': {'class': 1,  'mean': mean(ae_training), 'covariance': cov_matrix(ae_training), 'GMM': GMM(num_gaussians, cov_type, seed, ae_training)},
    'ah': {'class': 2,  'mean': mean(ah_training), 'covariance': cov_matrix(ah_training), 'GMM': GMM(num_gaussians, cov_type, seed, ah_training)},
    'aw': {'class': 3,  'mean': mean(aw_training), 'covariance': cov_matrix(aw_training), 'GMM': GMM(num_gaussians, cov_type, seed, aw_training)},
    'eh': {'class': 4,  'mean': mean(eh_training), 'covariance': cov_matrix(eh_training), 'GMM': GMM(num_gaussians, cov_type, seed, eh_training)},
    'er': {'class': 5,  'mean': mean(er_training), 'covariance': cov_matrix(er_training), 'GMM': GMM(num_gaussians, cov_type, seed, er_training)},
    'ei': {'class': 6,  'mean': mean(ei_training), 'covariance': cov_matrix(ei_training), 'GMM': GMM(num_gaussians, cov_type, seed, ei_training)},
    'ih': {'class': 7,  'mean': mean(ih_training), 'covariance': cov_matrix(ih_training), 'GMM': GMM(num_gaussians, cov_type, seed, ih_training)},
    'iy': {'class': 8,  'mean': mean(iy_training), 'covariance': cov_matrix(iy_training), 'GMM': GMM(num_gaussians, cov_type, seed, iy_training)},
    'oa': {'class': 9,  'mean': mean(oa_training), 'covariance': cov_matrix(oa_training), 'GMM': GMM(num_gaussians, cov_type, seed, oa_training)},
    'oo': {'class': 10, 'mean': mean(oo_training), 'covariance': cov_matrix(oo_training), 'GMM': GMM(num_gaussians, cov_type, seed, oo_training)},
    'uh': {'class': 11, 'mean': mean(uh_training), 'covariance': cov_matrix(uh_training), 'GMM': GMM(num_gaussians, cov_type, seed, uh_training)},
    'uw': {'class': 12, 'mean': mean(uw_training), 'covariance': cov_matrix(uw_training), 'GMM': GMM(num_gaussians, cov_type, seed, uw_training)}
}

def print_weights():
    for vowel, parameters in trained_parameters.items():
                gmm = parameters['GMM']
                weights = gmm.weights_
                means = gmm.means_
    
                print(f"Vowel: {vowel}")
                for i in range(len(weights)):
                    print(f"Component {i+1}:  Weight: {weights[i]} Mean: {means[i]}")
#print_weights()


"""------------------------ TESTING ----------------------------"""
def test(test_data):

    #Initializing lists and dictionary which stores the results of the test
    predicted_labels = []
    predicted_classes = []
    classification_counts = {vowel: 0 for vowel in trained_parameters}

    #Looping through all the test data
    for i in range(len(test_data)):

        max_prob = float('-inf')
        predicted_vowel = None

        #Looping through the different gaussians. Vowels are the keys, parameters are values
        for vowel, parameters in trained_parameters.items():

            ###      Single Gaussian model    ###
            mean = parameters['mean']
            cov = parameters['covariance']
            #prob = multivariate_normal(mean=mean, cov=cov).pdf(test_data[i])

            ###          GMM        ###
            gmm = parameters['GMM']
            prob = np.exp(gmm.score_samples([test_data[i]])[0])

            #Descision rule. The gauss which gives the largest probability is the class which is chosen
            if prob*P_w > max_prob:
                max_prob = prob*P_w
                class_num = parameters['class']
                predicted_vowel = vowel

        #Appending result to lists and dictionary
        predicted_labels.append(predicted_vowel)
        predicted_classes.append(class_num)
        classification_counts[predicted_vowel] += 1

    return predicted_labels, predicted_classes, classification_counts
"""---------------------------------------------------------------"""


def conf_matrix(pred):
     true = [i for i in range(1, 13) for _ in range(69)]

     cm = confusion_matrix(true, pred)
     error = 1 - accuracy_score(true, pred)

     return cm, error

def plot_conf_matrix(cm):
     
     ###Plotting the confusion matrix
     total = np.sum(cm)
     percent_occurrence = (cm / total) * 100
     plt.figure(figsize=(10, 8))
     sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False, linewidths=0.5, linecolor='grey', vmax=69, annot_kws={"fontsize": 14})
     phonetic_labels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']
     plt.xticks(np.arange(12) + 0.5, phonetic_labels, fontsize=14)
     plt.yticks(np.arange(12) + 0.5, phonetic_labels, fontsize=14)
     plt.title("Confusion Matrix for a GMM with 3 Gaussians \n and diagonal covariance matrix", fontsize=20)
     plt.xlabel("Predicted label", fontsize=16)
     plt.ylabel("True label", fontsize=16)
     plt.show()


def plot():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Define a scale factor for the data
    scale_factor = 0.1  # Adjust this value as needed
    
    # Scatter plot with scaled data
    ax.scatter(ae_training[:, 0] * scale_factor, ae_training[:, 1] * scale_factor, ae_training[:, 2] * scale_factor, label='ae')
    ax.scatter(ah_training[:, 0] * scale_factor, ah_training[:, 1] * scale_factor, ah_training[:, 2] * scale_factor, label='ah')
    ax.scatter(aw_training[:, 0] * scale_factor, aw_training[:, 1] * scale_factor, aw_training[:, 2] * scale_factor, label='aw')
    ax.scatter(eh_training[:, 0] * scale_factor, eh_training[:, 1] * scale_factor, eh_training[:, 2] * scale_factor, label='eh')
    ax.scatter(er_training[:, 0] * scale_factor, er_training[:, 1] * scale_factor, er_training[:, 2] * scale_factor, label='er')
    ax.scatter(ei_training[:, 0] * scale_factor, ei_training[:, 1] * scale_factor, ei_training[:, 2] * scale_factor, label='ei')
    ax.scatter(ih_training[:, 0] * scale_factor, ih_training[:, 1] * scale_factor, ih_training[:, 2] * scale_factor, label='ih')
    ax.scatter(iy_training[:, 0] * scale_factor, iy_training[:, 1] * scale_factor, iy_training[:, 2] * scale_factor, label='iy')
    ax.scatter(oa_training[:, 0] * scale_factor, oa_training[:, 1] * scale_factor, oa_training[:, 2] * scale_factor, label='oa')
    ax.scatter(oo_training[:, 0] * scale_factor, oo_training[:, 1] * scale_factor, oo_training[:, 2] * scale_factor, label='oo')
    ax.scatter(uh_training[:, 0] * scale_factor, uh_training[:, 1] * scale_factor, uh_training[:, 2] * scale_factor, label='uh')
    ax.scatter(uw_training[:, 0] * scale_factor, uw_training[:, 1] * scale_factor, uw_training[:, 2] * scale_factor, label='uw')

    plt.legend()
    plt.show()

#plot()

def histogram():
    fig, axs = plt.subplots(3, 1)
    features = ['f1', 'f2', 'f3']
    # ae, ah, aw, eh, er, ei, ih, iy, oa, oo, uh, uw
    bin_width = 40
    for i in range(3):
        ax = axs[i]
        bins = np.arange(0, 3700 + bin_width, bin_width)
        sns.histplot(ae_training[:, i], bins=bins, label=f'Formant nr.{i+1} er', alpha=0.5, ax=ax, kde=True)
        sns.histplot(ah_training[:, i], bins=bins, label=f'Formant nr.{i+1} ah', alpha=0.5, ax=ax, kde=True)
        sns.histplot(aw_training[:, i], bins=bins, label=f'Formant nr.{i+1} aw', alpha=0.5, ax=ax, kde=True)
        sns.histplot(eh_training[:, i], bins=bins, label=f'Formant nr.{i+1} eh', alpha=0.5, ax=ax, kde=True)
        sns.histplot(er_training[:, i], bins=bins, label=f'Formant nr.{i+1} er', alpha=0.5, ax=ax, kde=True)
        sns.histplot(ei_training[:, i], bins=bins, label=f'Formant nr.{i+1} ei', alpha=0.5, ax=ax, kde=True)
        sns.histplot(ih_training[:, i], bins=bins, label=f'Formant nr.{i+1} ih', alpha=0.5, ax=ax, kde=True)
        sns.histplot(iy_training[:, i], bins=bins, label=f'Formant nr.{i+1} iy', alpha=0.5, ax=ax, kde=True)
        sns.histplot(oa_training[:, i], bins=bins, label=f'Formant nr.{i+1} oa', alpha=0.5, ax=ax, kde=True)
        sns.histplot(oo_training[:, i], bins=bins, label=f'Formant nr.{i+1} oo', alpha=0.5, ax=ax, kde=True)
        sns.histplot(uh_training[:, i], bins=bins, label=f'Formant nr.{i+1} uh', alpha=0.5, ax=ax, kde=True)
        sns.histplot(uw_training[:, i], bins=bins, label=f'Formant nr.{i+1} uw', alpha=0.5, ax=ax, kde=True)

        ax.set_xlabel(f'Frequency [Hz]', fontsize=16)
        ax.set_ylabel(f'Count', fontsize=12)
        #ax.legend(loc='upper right')
    plt.show()

histogram()

def plot_gaussian():
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    features = ['f1', 'f2', 'f3']
    bin_width = 60
    for i in range(3):
        ax = axs[i]
        
        # Plot Gaussian distributions
        for phoneme in ['er', 'ih']:
            params = trained_parameters[phoneme]
            gmm = params['GMM']
            mean = params['mean'][i]
            covariance = params['covariance'][i, i]  # Assuming diagonal covariance matrix
            gaussian = multivariate_normal(mean=mean, cov=covariance)
            x = np.linspace(0, 4000, 1000)
            y = gaussian.pdf(x)
            ax.plot(x, y, label=f'{phoneme} Gaussian', linestyle='--')

        ax.set_xlabel(f'Frequency [Hz]', fontsize=16)
        ax.legend(loc='upper right')
    plt.show()

#plot_gaussian()

###Testing###
print(test(test_data)[0], '\n')
print(test(test_data)[1], '\n')
print(test(test_data)[2], '\n')

predicted = test(test_data)[1]
conf_mat = conf_matrix(predicted)[0]
error = conf_matrix(predicted)[1]

print("Confusion matrix: \n", conf_mat)
print("Error rate: \n", error)

plot_conf_matrix(conf_matrix(predicted)[0])

#print("ae training set: ", ae_training)
#print("Mean value vector: ", mean(ae_training))
#print("Covariance matrix: ", cov_matrix(ae_training))
#print(multivariate_gaussian(mean(ae_training), cov_matrix(ae_training)))
