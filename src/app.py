from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', delimiter=';')

# Step 2: Encode the target variable 'y' (Yes -> 1, No -> 0)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Step 3: Use only numeric features for logistic regression
X_numeric = df.select_dtypes(include=['int64', 'float64'])
y = df['y']

# Step 4: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.3, random_state=42)

# Step 5: Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Step 7: Evaluate the model
y_pred = logreg.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 8: Check for class imbalance
print("\nTarget Variable Distribution:")
print(df['y'].value_counts(normalize=True))

# Step 9: Perform cross-validation
cross_val_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("\nCross-Validation Scores (5-fold):")
print(cross_val_scores)

# Cross-Validation results summary
print(f"Mean accuracy from cross-validation: {cross_val_scores.mean()}")
