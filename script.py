from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from Class import *
import os
from itertools import product

# List of datasets to process
datasets = [
    "car evaluation",
    "Iris",
    # "Statlog (Australian Credit Approval)",
    "Contraceptive Method Choice",
    "tic tac"
]

# Different RACER configurations to test
param_grid = {
    "alpha": [0.95],  
    "gamma": [0.6],  
    "apriori": [True],  
    "feature_train": [True, False],  
    "feature_class": [True, False],  
    "support_threshold": [0.1,0.01],  
    "fitness_threshold": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
}

# Generate all combinations of parameters dynamically
configs = [
    dict(zip(param_grid.keys(), values)) 
    for values in product(*param_grid.values())
]
manual = {
    "alpha": 0.90,
    "gamma": 0.7,
    "apriori": False,
    "feature_train": False,
    "feature_class": False,
    "support_threshold": 0,
    "fitness_threshold": 0
}

configs.insert(0,manual)

# Output Excel file
output_file = "RACER_Final_Results.xlsx"

# If file exists, load previous results to prevent overwriting
if os.path.exists(output_file):
    existing_df = pd.read_excel(output_file)
    results_list = existing_df.values.tolist()  # Convert back to list
else:
    results_list = []

# Process each dataset with multiple configurations
for dbName in datasets:
    # try:
        print(f"\nProcessing dataset: {dbName}...")

        # Load ARFF file
        filePath = f"C:\\Users\\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\dataSet\\{dbName}\\{dbName}.arff"
        data, meta = arff.loadarff(filePath)
        dataTypes = meta.types()
        dataSet = pd.DataFrame(data).values

        # Preprocess the dataset
        preprocessor = RACERPreprocessor()
        X, y = preprocessor.fit_transform(dataSet, dataTypes)

        # Cross-validation setup
        n_splits1 = 10
        kf = KFold(n_splits=n_splits1, random_state=1, shuffle=True)

        # Iterate over different configurations
        for config in configs:
            alpha = config["alpha"]
            gamma = config["gamma"]
            support_threshold = config["support_threshold"]
            fitness_threshold = config["fitness_threshold"]
            feature_apriori = config["apriori"]
            feature_class = config["feature_class"]
            feature_train = config["feature_train"]

            total_accuracy = 0
            total_numOfRules = 0
            total_numOfAprioriRule = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = y[train_index], y[test_index]

                # Initialize and train RACER with the current configuration
                racer = RACER(alpha=alpha, gamma=gamma, suppress_warnings=True,
                              feature_apriori=feature_apriori,
                              feature_class=feature_class, feature_train=feature_train,
                              support_treshhold=support_threshold, fitness_treshhold=fitness_threshold)
                
                aprioriRules = racer.fit(X_train, Y_train)
                racer.reduceRules()

                # Compute accuracy and number of rules
                score = racer.score(X_test, Y_test)
                rules = racer.getNumOfRules()

                total_accuracy += score
                total_numOfRules += rules

                if(type(aprioriRules) != bool): 
                    total_numOfAprioriRule.append(aprioriRules)
                else:
                     total_numOfAprioriRule.append([0])

            # Store final averaged results for this configuration
            avg_accuracy = total_accuracy / n_splits1
            avg_rules = total_numOfRules / n_splits1
            avg_apriori = [x / n_splits1 for x in list(map(sum, zip(*total_numOfAprioriRule)))]


            results_list.append([dbName, alpha, gamma, feature_apriori,
                                feature_class,feature_train, support_threshold,
                                fitness_threshold, avg_accuracy, avg_rules, avg_apriori])

            # print(f"Final Results for {dbName} (α={alpha}, γ={gamma}, sup={support_threshold}, fit={fitness_threshold}):")
            # print(f"  - Average Accuracy: {avg_accuracy*100:.2f}%")
            # print(f"  - Average Number of Rules: {avg_rules}\n")

            # Convert results to DataFrame and save iteratively
            df_results = pd.DataFrame(results_list, columns=["Dataset", "Alpha", "Gamma","feature_apriori","feature_class","feature_train", "Support Threshold", "Fitness Threshold", "Final Accuracy", "Final Number of Rules","avg_apriori"])
            df_results.to_excel(output_file, index=False)
            print(f"Results saved to {output_file}")

    # except Exception as e:
    #     print(f"Error processing {dbName}: {e}")

print("\nAll datasets and configurations processed. Final results saved.")
