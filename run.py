from scipy.io import arff
import pandas as pd
from AcceleRACER import *

def run(path):
    accuracy = 0;
    numOfRules = 0;    
    for i in range(1, 11):
        foldPath = path + "\\fold-{0}".format(i);
        trainData, trainMetaData = arff.loadarff(foldPath+"\\train.arff");
        testData, testMetaData = arff.loadarff(foldPath+"\\test.arff");        
        dataTypes = trainMetaData.types();        
        
        dataSet = np.concatenate((testData, trainData), axis=0);        
        
        dataSet = pd.DataFrame(dataSet).values;        
        
        preprocessor = RACERPreprocesser();        
        X, y = preprocessor.fit_transform(dataSet, dataTypes);        
        X_test = X[0:testData.shape[0], :];
        y_test = y[0:testData.shape[0], :];
        X = X[testData.shape[0]:, :];
        y = y[testData.shape[0]:, :];        
        
        racer = RACER(alpha = 0.95, gamma = 0.6, suppress_warnings = True);
        racer.fit(X,y);        

        racer.reduceRules();

        score = racer.score(X_test, y_test);
        rules = racer.getNumOfRules();
        numOfRules += rules;
        accuracy += score;

        print("run {0}:".format(i));
        print("accuracy : {0}".format(score));
        print("numer of rules : {0}".format(rules));
        print("\n");
        

    print("#####################\n");
    print("\tfinal reslut for 10 runs : ")
    print("\tfinal rules : {0}".format(numOfRules/10));
    print("\tfinal accuracy : {:.2f}".format((accuracy/10)*100));    
    print("\n#####################");
     



# Statlog (Australian Credit Approval)
# Iris
# Contraceptive Method Choice
dbName = "Iris"
filePath = "C:\\Users\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\dataSet\\{0}\\{0}".format(dbName, dbName);   
run(filePath);


    
