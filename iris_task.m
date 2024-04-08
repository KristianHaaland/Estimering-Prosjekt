
% There are 3 classes of irises, each with matrices conatining 50 examples of each plant.
% Each plant has 4 features (50x4).

%Class 1 = Iris Setosa
%Class 2 = Iris-versicolor
%Class 3 = Iris-virginica

Class_1 = readtable('class_1');
Class_2 = readtable('class_2');
Class_3 = readtable('class_3');
All_classes = readtable('iris.data');


%Separating into training set and testing set
Class_1_training = Class_1(1:30,:);
Class_1_testing = Class_1(31:50,:);

Class_2_training = Class_2(1:30,:);
Class_2_testing = Class_32(31:50,:);

Class_3_training = Class_3(1:30,:);
Class_3_testing = Class_3(31:50,:);

%Histograms (distributions) of features in the 3 classes 
feature = 1;
feature_histogram = [table2array(Class_1(:,feature));table2array(Class_2(:,feature));table2array(Class_3(:,feature))];
histogram(feature_histogram);