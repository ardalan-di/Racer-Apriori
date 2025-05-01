from scipy.io import arff
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from Class import *


"""
# Convert the ARFF data to a structured NumPy array
data_array = np.array(data.tolist(), dtype=data.dtype)
print("Data Array Shape:", data_array.shape)

# Separate features (X) and labels (y)
X = data_array[:,:-1] # Features
y = data_array[:,-1]  # Labels

# Modify the dataset path below to test other ARFF files.
"""

#  Statlog (Australian Credit Approval)
#  Iris
#  Contraceptive Method Choice
#  car evaluation
#  tic tac
# dbName = "3-pyrim"
dbName = "Iris"
filePath = "C:\\Users\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\dataSet\\{0}\\{0}.arff".format(dbName, dbName);   
# filePath = f"C:\\Users\\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\data\\{dbName}.arff"

data, meta = arff.loadarff(filePath)
dataTypes = meta.types();
dataSet = pd.DataFrame(data).values;
preprocessor = RACERPreprocessor();
X, y = preprocessor.fit_transform(dataSet, dataTypes);                 
accuracy = 0 ;
numOfRules = 0;
n_splits1 = 10;
kf = KFold(n_splits=n_splits1 , random_state=1 , shuffle=True)
i=0
for train_index, test_index in kf.split(X):
    X_train , X_test , Y_train , Y_test = [] , [] , [] , []
    X_train , X_test = X[train_index] , X[test_index]
    Y_train , Y_test = y[train_index] , y[test_index]
    racer = None
    racer = RACER(alpha = 0.99, gamma = 0.6, suppress_warnings = True,
            feature_apriori=True,
            feature_class=False, feature_train=True,
            support_treshhold=0.01, fitness_treshhold=0.50
          );
    racer.fit(X_train, Y_train);
    racer.reduceRules();
    score = racer.score(X_test, Y_test);
    rules = racer.getNumOfRules();
    numOfRules += rules;
    accuracy += score;
    print("Results for the fold {0}:".format(i));
    print("Accuracy: {0}".format(score));
    print("Number of rules generated: {0}".format(rules));
    print("\n");
    i=i+1


print("#####################\n");
print("\tFinal results for {0} folds:".format(n_splits1))
print("\tTotal number of rules generated: {0}".format(numOfRules/10));
print("\tFinal accuracy in percentage: {:.2f}".format((accuracy/10)*100));    
print("\n#####################");