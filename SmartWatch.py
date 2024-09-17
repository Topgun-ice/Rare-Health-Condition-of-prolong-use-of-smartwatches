import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.model import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Dropout


# Set random seed for reproducibility
np.random.seed(42)

# Number of samples in the dataset
num_samples = 10000

# Generate randomized smartwatch data
data = {
    'heart_rate': np.random.normal(loc=70, scale=10, size=num_samples),  # Average heart rate (BPM)
    'steps': np.random.randint(500, 20000, size=num_samples),            # Steps taken in a day
    'sleep_hours': np.random.uniform(4, 10, size=num_samples),           # Hours of sleep
    'oxygen_saturation': np.random.uniform(90, 100, size=num_samples),   # Oxygen saturation level (%)
    'calories_burned': np.random.uniform(1500, 3500, size=num_samples),  # Calories burned in a day
    'active_minutes': np.random.uniform(10, 180, size=num_samples),      # Active minutes
    'age': np.random.randint(18, 80, size=num_samples),                  # Age of the person
    'weight': np.random.normal(loc=70, scale=15, size=num_samples),      # Weight in kg
    'height': np.random.normal(loc=170, scale=10, size=num_samples),     # Height in cm
    'gender': np.random.choice(['Male', 'Female'], size=num_samples),    # Gender
}

# Generate the target variable (rare condition, 1 for yes, 0 for no)
# Assuming the condition affects 1% of the population
data['rare_condition'] = np.random.choice([0, 1], size=num_samples, p=[0.99, 0.01])

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Show the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('smartwatch_health_data.csv', index=False)

print("Randomized dataset generated and saved to 'smartwatch_health_data.csv'.")

# Load the dataset
df = pd.read_csv('smartwatch_health_data.csv')

# Check the first few rows of the dataset
print(df.head())

# Explore the data
print(df.info())

# You can now proceed to preprocess the data, train your machine learning models, etc.

# Load the dataset
data = pd.read_csv('smartwatch_health_data.csv')
print(data.head())

# Explore the dataset
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualize the distribution of the target variable
sns.countplot(x='rare_condition', data=data)

plt.title('Condition Distribution')
plt.show()

# Features and target
X = data.drop(columns=['rare_condition'])
y = data['rare_condition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM (assuming a sequence length of 30 time steps)
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define LSTM model
model = Sequential()

# LSTM layers
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=64))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model’s performance using the test dataset.
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Make predictions
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the trained model
model.save('smartwatch_health_condition_model.h5')
