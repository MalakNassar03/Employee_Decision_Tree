#Malak Nassar
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the Employee.csv dataset
data = pd.read_csv("Employee.csv")

# Encode categorical variables using LabelEncoder
label_encoders = {}
for column in data.columns:
    if data[column].dtype == "object":
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Specify the target variable column name
target_column_name = "LeaveOrNot"

# Split the dataset into features (X) and the target variable (y)
X = data.drop(target_column_name, axis=1)
y = data[target_column_name]

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the decision tree classifier
clf.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(15, 8))
plot_tree(clf, filled=True, feature_names=list(X.columns), class_names=[str(cls) for cls in clf.classes_])
plt.show()
