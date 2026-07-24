import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the crop recommendation dataset
data = pd.read_csv("data/Crop_recommendation.csv")

# Separate input features and target
X = data.drop("label", axis=1)
y = data["label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2%}")

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
