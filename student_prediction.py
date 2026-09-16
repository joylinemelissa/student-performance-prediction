import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("dataset.csv")

# Input data
X = data[["study_hours", "attendance", "previous_marks", "assignment_score"]]

# Output we want to predict
y = data["final_score"]

print("Data is ready for Machine Learning!")

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

# Create the Machine Learning model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Model trained successfully!")
# Make a prediction for one student
prediction = model.predict([[7, 85, 80, 78]])

print("Predicted final score:", prediction[0])

# Make predictions using the testing data
y_pred = model.predict(X_test)

print("Actual scores:", list(y_test))
print("Predicted scores:", list(y_pred))

from sklearn.metrics import mean_absolute_error, r2_score

# Calculate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

model.predict([[7, 85, 80, 78]])

# Get student details from the user
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_marks = float(input("Enter previous marks: "))
assignment_score = float(input("Enter assignment score: "))

# Make prediction
prediction = model.predict([[
    study_hours,
    attendance,
    previous_marks,
    assignment_score
]])

print("Predicted final score:", round(prediction[0], 2))

import matplotlib.pyplot as plt

# Compare actual and predicted scores
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Scores")

plt.show()