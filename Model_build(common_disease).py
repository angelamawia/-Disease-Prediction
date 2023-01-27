# Load libraries
from pandas import read_csv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Load the data
Test_dataset_path = 'disease dataset/Testing.csv'
Train_dataset_path = 'disease dataset/Training.csv'

Train_Data = read_csv(Train_dataset_path)
Test_Data = read_csv(Test_dataset_path)

# Data cleaning
Train_Data = Train_Data.dropna(axis=1)  # remove columns that contain missing values.
Train_Data = Train_Data.dropna(axis=0)  # remove rows that contain missing values.

Test_Data = Test_Data.dropna(axis=1)  # remove columns that contain missing values.
Test_Data = Test_Data.dropna(axis=0)  # remove rows that contain missing values.

# feature extraction
feature_data = Train_Data.iloc[:, :-1].values
target_data = Train_Data.iloc[:, -1].values

# Split the dataset into 'feature-data' and 'target-data

X_train, X_validation, Y_train, Y_validation = train_test_split(feature_data, target_data, test_size=0.20, random_state=1)

# Build Models
print('__model---------------------')


model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


print('------- x validation ------------')
print(X_validation)
print('======== prediction =============')
print(predictions)


# Evaluate predictions
print('\n\n\n\n__model accuracy_score---------------------\n')
print(accuracy_score(Y_validation, predictions))
print('__model- confusion_matrix--------------------\n')
print(confusion_matrix(Y_validation, predictions))
print('__model classification_report---------------------\n')

print(classification_report(Y_validation, predictions))


# saving the model


joblib.dump(model, 'model-disease-prediction.joblib')
