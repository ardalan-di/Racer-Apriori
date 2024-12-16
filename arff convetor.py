import os
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


dbName = "car evaluation"
filePath = "C:\\Users\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\dataSet\\{0}\\{0}.arff".format(dbName);  

# Load the ARFF file into a DataFrame
data, meta = arff.loadarff(filePath)
df = pd.DataFrame(data)

df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Separate features (X) and target (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Create a folder to store the train and test files for each fold
output_folder = "C:\\Users\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\dataSet\\{0}\\{0}".format(dbName);  
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize StratifiedKFold to ensure balanced splits
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# Iterate through each fold
fold = 1
for train_index, test_index in kf.split(X,y):
    # Split the data into train and test sets for the fold
    X_fold = X.iloc[test_index]
    y_fold = y.iloc[test_index]

    # Now, for each fold, we split the fold into 90% train and 10% test within that fold
    fold_train_size = int(0.9 * len(X_fold))
    X_fold_train = X_fold.iloc[:fold_train_size]
    y_fold_train = y_fold.iloc[:fold_train_size]
    
    X_fold_test = X_fold.iloc[fold_train_size:]
    y_fold_test = y_fold.iloc[fold_train_size:]
    
    # Combine features and target for train and test datasets
    train_data = pd.concat([X_fold_train, y_fold_train], axis=1)
    test_data = pd.concat([X_fold_test, y_fold_test], axis=1)
    
    # Save the train and test files in ARFF format
    # Save train data
    
    with open("C:\\Users\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\dataSet\\{0}\\{0}\\fold-{2}\\train.arff".format(dbName,dbName,fold), 'w') as f:
        # Write ARFF header
        f.write('@relation car\n\n')
        for col in X.columns:
            f.write(f'@attribute {col} {{' + ', '.join(X[col].unique().astype(str)) + '}\n')
        f.write('@attribute class {unacc, acc, good, vgood}\n\n')
        f.write('@data\n')
        
        # Write data
        for index, row in train_data.iterrows():
            f.write(','.join(map(str, row)) + '\n')
    
    # Save test data
    with open("C:\\Users\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\dataSet\\{0}\\{0}\\fold-{2}\\test.arff".format(dbName,dbName,fold), 'w') as f:
        # Write ARFF header
        f.write('@relation car\n\n')
        for col in X.columns:
            f.write(f'@attribute {col} {{' + ', '.join(X[col].unique().astype(str)) + '}\n')
        f.write('@attribute class {unacc, acc, good, vgood}\n\n')
        f.write('@data\n')
        
        # Write data
        for index, row in test_data.iterrows():
            f.write(','.join(map(str, row)) + '\n')
    
    print(f"Fold {fold}: Train and Test ARFF files saved.")
    fold += 1

print(f"All folds saved in '{output_folder}' directory.")
