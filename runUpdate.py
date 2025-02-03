import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from AcceleRACER import RACER, RACERPreprocessor

# Path to the ARFF file
arff_file_path = "C:\\Users\\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\dataSet\\car evaluation\\car evaluation.arff"

# Load ARFF data
data, meta = arff.loadarff(arff_file_path)

# Convert ARFF to Pandas DataFrame
df = pd.DataFrame(data)

# Decode bytes to strings for categorical attributes
for col in df.select_dtypes([object]).columns:
    df[col] = df[col].str.decode('utf-8')

# Convert to categorical data type
X = df.drop(columns=['Class']).astype('category')
Y = df[['Class']].astype('category')

# print(X)
# print(Y)
# Apply RACERPreprocessor
X, Y = RACERPreprocessor().fit_transform(X, Y)

# print(X)
# print(Y)

# print(X.shape[1])
# print(Y.shape[1])

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.1)

# Initialize and train RACER
racer = RACER(alpha=0.95,feature_class=True,feature_train=True,support_treshhold=0.1,fitness_treshhold=0.99,feature_no_fitness_change=False,suppress_warnings=False, benchmark=True)
racer.fit(X_train, Y_train)

# Print RACER model score
print(racer.score(X_test, Y_test))
