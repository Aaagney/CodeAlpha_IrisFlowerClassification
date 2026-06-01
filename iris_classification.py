from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print(f"Model Accuracy: {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred))
# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Sample prediction
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("Predicted Flower:", iris.target_names[prediction][0])