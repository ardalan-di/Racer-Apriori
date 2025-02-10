from scipy.io import arff
import pandas as pd
import numpy as np
from AcceleRACER import *

def load_arff_data(path, fold):
    """Loads train and test data for a given fold."""
    foldPath = f"{path}\\fold-{fold}"
    trainData, trainMetaData = arff.loadarff(f"{foldPath}\\train.arff")
    testData, testMetaData = arff.loadarff(f"{foldPath}\\test.arff")
    return trainData, trainMetaData, testData, testMetaData

def run(path):
    accuracy = 0;
    numOfRules = 0;    
    for i in range(1, 11):
        trainData, trainMetaData, testData, testMetaData = load_arff_data(path, i)       
        dataTypes = trainMetaData.types();        
        
        dataSet = np.concatenate((testData, trainData), axis=0)
        df = pd.DataFrame(dataSet, columns=trainMetaData.names())

        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].str.decode('utf-8')
        
        X = df.drop(columns=['class']).astype('category')
        Y = df[['class']].astype('category')
       
        
        preprocessor = RACERPreprocessor()
        X, Y = preprocessor.fit_transform(X, Y)

        # Split train and test sets
        test_size = testData.shape[0]
        X_test, Y_test = X[:test_size, :], Y[:test_size, :]
        X_train, Y_train = X[test_size:, :], Y[test_size:, :]       
        
        racer = RACER(
            alpha=0.95, feature_class=True, feature_train=True,
            support_treshhold=0.1, fitness_treshhold=0.99,
            suppress_warnings=False, benchmark=True
        )
        racer.fit(X_train, Y_train)        

        score = racer.score(X_test, Y_test)
        accuracy += score

        print("run {0}:".format(i));
        print("accuracy : {0}".format(score));
        print("\n");
        

    print("#####################\n");
    print("\tfinal reslut for 10 runs : ")
    print("\tfinal accuracy : {:.2f}".format((accuracy/10)*100));    
    print("\n#####################");
     



# Statlog (Australian Credit Approval)
# Iris
# Contraceptive Method Choice
# car evaluation
# tic tac
dbName = "car evaluation"
filePath = "C:\\Users\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\dataSet\\{0}\\{0}".format(dbName, dbName);   
run(filePath);


    
