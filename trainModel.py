import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Loading the dataset
data = pd.read_csv('global_air_pollution_dataset.csv')

# Checking correlation between AQI Value and PM2.5 AQI Value
correlation = data['AQI Value'].corr(data['PM2.5 AQI Value'])
print(f"Correlation between AQI Value and PM2.5 AQI Value: {correlation:.4f}")

# Adding a feature for maximum AQI value among pollutants
data['Max AQI'] = data[['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']].max(axis=1)

# Selecting features and target
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value', 'Max AQI']
target = 'AQI Category'

# Encoding the target variable
le = LabelEncoder()
data['AQI Category Encoded'] = le.fit_transform(data[target])

# Preparing X (features) and y (target)
X = data[features]
y = data['AQI Category Encoded']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Performing 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Training the model on the full training set
model.fit(X_train, y_train)

# Evaluating the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Saving the model and label encoder to a .pkl file
joblib.dump({'model': model, 'label_encoder': le}, 'air_pollution_model.pkl')
print("Model and label encoder saved to air_pollution_model.pkl")