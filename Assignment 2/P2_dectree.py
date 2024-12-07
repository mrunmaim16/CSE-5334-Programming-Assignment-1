# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load and preprocess dataset
nba_stats = pd.read_csv('nba_stats.csv')

# One-hot encode categorical columns (e.g., 'Tm')
nba_stats_encoded = pd.get_dummies(nba_stats, columns=['Tm'])

# Define feature set and target variable
X = nba_stats_encoded.drop(columns=['Pos'])
y = nba_stats['Pos']

# Split the data into training and validation sets (80%-20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Configure and train the RandomForest model
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=0
)
rf_clf.fit(X_train, y_train)

# Predict and assess accuracy on both training and validation sets
y_train_pred = rf_clf.predict(X_train)
y_val_pred = rf_clf.predict(X_val)
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

# Display results for Task 1
print("Task 1: Train-Test Split and Model Training")
print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print("Confusion Matrix - Training Set:")
print(confusion_matrix(y_train, y_train_pred))
print("Confusion Matrix - Validation Set:")
print(confusion_matrix(y_val, y_val_pred))

# Task 2: Evaluate on Dummy Test Dataset
dummy_test_data = pd.read_csv('dummy_test.csv')

# One-hot encode and scale dummy test data
dummy_test_data_encoded = pd.get_dummies(dummy_test_data, columns=['Tm'])
dummy_test_data_encoded = dummy_test_data_encoded.reindex(columns=X.columns, fill_value=0)
dummy_test_data_encoded = scaler.transform(dummy_test_data_encoded)

# Predict and calculate accuracy for dummy dataset
y_dummy = dummy_test_data['Pos']
y_dummy_pred = rf_clf.predict(dummy_test_data_encoded)
dummy_accuracy = accuracy_score(y_dummy, y_dummy_pred)

# Display results for Task 2
print("\nTask 2: Test on Dummy Dataset")
print(f"Dummy Test Accuracy: {dummy_accuracy}")
print("Confusion Matrix - Dummy Test Set:")
print(confusion_matrix(y_dummy, y_dummy_pred))

# Task 3: Perform 10-Fold Cross-Validation
cv_scores = cross_val_score(rf_clf, X, y, cv=10, n_jobs=-1)

# Display results for Task 3
print("\nTask 3: 10-Fold Cross-Validation")
print("Cross-Validation Accuracies for each fold:")
for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score}")
print(f"Average Cross-Validation Accuracy: {cv_scores.mean()}")
