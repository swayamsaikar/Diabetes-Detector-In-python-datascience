import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# !reading the data
df = pd.read_csv('diabetes.csv')

# !making groups of factors and result
input_factors = df[["Glucose", "BloodPressure", "Age"]]
result = df["Outcome"]

# !splitting the whole dataset into train and test sets
train_factors, test_factors, train_result, test_result = train_test_split(
    input_factors, result, test_size=0.25, random_state=0)

# !we are using standardScaler() to standardize and make all the values in train_factors and test_factors, as the same unit
# !initializing the standardScaler()
sc = StandardScaler()

train_factors = sc.fit_transform(train_factors)
test_factors = sc.fit_transform(test_factors)

# !Training our model with all the train sets
lr = LogisticRegression(random_state=0)
lr.fit(train_factors, train_result)

# !lr.predict() <- will take the factors that are there for testing our model as INPUT
predicted_result = lr.predict(test_factors)

# ! showing the predicted results list
print(f"predicted_result -> {predicted_result}")

# ! we are using the accuracy_score module from sklearn which will take the actual testing results(test_result) and the predicted results(predicted_result) and then calculate the accuracy of our model
score = accuracy_score(test_result, predicted_result)
score_in_percent = str(score)[2:4]
print(f"Accuracy -> {score_in_percent}%")
