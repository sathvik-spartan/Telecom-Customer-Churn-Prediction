import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout # type: ignore

# Load the raw dataset
raw_data = pd.read_csv("telecom_customer_churn_dataset.csv")

# Load the binary dataset
binary_data = pd.read_csv("telecom_customer_churn_dataset_binary.csv")

# Read the first few rows of both datasets
df1 = pd.read_csv("telecom_customer_churn_dataset.csv")
df2 = pd.read_csv("telecom_customer_churn_dataset_binary.csv")


### --- EXPLORATORY DATA ANALYSIS (EDA) ---
print("Checking for missing values:\n", raw_data.isnull().sum())

# Summary statistics
print("\nDataset Summary:\n", raw_data.describe())

# Convert categorical variables to numerical using Label Encoding
le = LabelEncoder()
for col in raw_data.select_dtypes(include=['object']).columns:
    if col == "Churn":
        continue
    raw_data[col] = le.fit_transform(raw_data[col])

# Visualizing correlation matrix
plt.figure(figsize=(12, 6))
sns.heatmap(raw_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Distribution of churn
sns.countplot(x="Churn Label", data=df1)
plt.title("Churn Distribution")
plt.show()

### --- DATA PREPROCESSING ---
X = raw_data.drop(columns=["Churn Label"])  # Features
y = raw_data["Churn Label"]  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### --- CLASSIFICATION MODELS ---

# 1. Naïve Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("\nNaïve Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# 2. Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# 3. CNN Model (Convolutional Neural Network)
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation="relu", input_shape=(X_train_cnn.shape[1], 1)),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))
cnn_pred = (cnn_model.predict(X_test_cnn) > 0.5).astype("int32")
print("\nCNN Accuracy:", accuracy_score(y_test, cnn_pred))

# 4. Deep Learning Model (Fully Connected Neural Network)
dl_model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

dl_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
dl_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
dl_pred = (dl_model.predict(X_test) > 0.5).astype("int32")
print("\nDeep Learning Model Accuracy:", accuracy_score(y_test, dl_pred))

### --- BINARY DATASET: CAUSE AND EFFECT DECISION TREE ---
X_bin = binary_data.drop(columns=["Churn Label"])
y_bin = binary_data["Churn Label"]

# Convert categorical features to numerical using Label Encoding
for col in X_bin.select_dtypes(include=['object']).columns:
    X_bin[col] = le.fit_transform(X_bin[col]) # Applying Label Encoding to categorical features in X_bin

dt_bin_model = DecisionTreeClassifier(random_state=42)
dt_bin_model.fit(X_bin, y_bin)

# Print Decision Tree Rules
tree_rules = export_text(dt_bin_model, feature_names=list(X_bin.columns))
print("\nDecision Tree Rules:\n", tree_rules)

### --- CLUSTERING TO IDENTIFY HIGH-RISK CHURN GROUPS ---
# Remove class labels
X_cluster = raw_data.drop(columns=["Churn Label"])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
raw_data["Cluster"] = kmeans.fit_predict(X_cluster)

# Visualizing Clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x=raw_data["Tenure"], y=raw_data["Monthly Charges"], hue=raw_data["Cluster"], palette="viridis")
plt.title("Tenure vs Monthly Charges Clustering")
plt.show()
