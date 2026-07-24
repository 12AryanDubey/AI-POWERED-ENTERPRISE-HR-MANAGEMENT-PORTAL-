import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("⏳ Generating synthetic HR dataset...")

# Step 1: Generate realistic HR data for 1000 employees
np.random.seed(42)
n_samples = 1000

data = {
    'satisfaction_level': np.random.uniform(0.1, 1.0, n_samples),
    'last_evaluation': np.random.uniform(0.3, 1.0, n_samples),
    'number_project': np.random.randint(2, 7, n_samples),
    'average_montly_hours': np.random.randint(90, 310, n_samples),
    'time_spend_company': np.random.randint(1, 10, n_samples),
    'work_accident': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
    'promotion_last_5years': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
    'salary_level': np.random.choice([1, 2, 3], size=n_samples, p=[0.4, 0.4, 0.2])  # 1: Low, 2: Medium, 3: High
}

df = pd.DataFrame(data)

# Logic rule: Employees with low satisfaction, high hours, and no promotion are likely to leave
df['attrition'] = (
    (df['satisfaction_level'] < 0.4) & 
    (df['average_montly_hours'] > 200) & 
    (df['promotion_last_5years'] == 0)
).astype(int)

# Introduce a bit of noise
noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
df.loc[noise_indices, 'attrition'] = 1 - df.loc[noise_indices, 'attrition']

X = df.drop('attrition', axis=1)
y = df['attrition']

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale features and train Random Forest Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Check Accuracy
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Trained Successfully! Accuracy: {acc * 100:.2f}%")

# Step 4: Save Model and Scaler locally in 'models/' folder
os.makedirs('models', exist_ok=True)

with open('models/attrition_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(" Saved 'attrition_model.pkl' and 'scaler.pkl' in the 'models/' folder.")