import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
file_path = "Naan mudhalvan Dataset (1).xlsx"  # Change name if different
df = pd.read_excel(file_path)

# Preview
print("Preview:\n", df.head())

# Select features and target
features = ['HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
            'AnyHealthcare', 'NoDocbcCost']
target = 'HeartDiseaseorAttack'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "disease_model.pkl")
print("✅ Model saved as disease_model.pkl")
