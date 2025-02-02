import pandas as pd
# from RACER import RACER, RACERPreprocessor
from sklearn.model_selection import train_test_split
from AcceleRACER import *



df = pd.read_csv(
    "C:\\Users\\Hkr\\Desktop\\bachelor project\\racerCode\\Racer-Apriori\\dataSet\\car evaluation\\car.data",
)

X = df.drop(columns=['class']).astype('category')
Y = df[['class']].astype('category')

X, Y = RACERPreprocessor().fit_transform(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=1, test_size=0.1)

racer = RACER(alpha=0.998, suppress_warnings=False, benchmark=True)
racer.fit(X_train, Y_train)
print(racer.score(X_test, Y_test))