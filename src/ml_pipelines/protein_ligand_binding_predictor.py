"""
Protein–Ligand Binding Prediction Pipeline
Predicts whether a protein binds a ligand based on structural features.

NOTE- THIS IS NOT YET IMPLEMENTED, IT IS A WIP. For me to get this to work with limited computing power, and have it extremely accurate is proving to be
    very difficult.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
import joblib

# Paths
DATA_PATH = "master_data/rcsb_pdb/RCSB_PDB_Macromolecular_Structure_Dataset_with_Structural_Features.csv"
MODEL_OUT = "models/protein_ligand_binding_model.keras"
SCALER_OUT = "models/protein_ligand_binding_scaler.pkl"

os.makedirs("models", exist_ok=True)

# Load dataset 
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows")

# Select features
feature_cols = [
    "Number of Residues",
    "Number of Chains",
    "Helix",
    "Sheet",
    "Coil",
    "Molecular Weight per Deposited Model"
]

# Drop rows missing structural features
df = df.dropna(subset=feature_cols)
print(f"After dropping missing structural features: {df.shape[0]} rows")

# Binary target: 1 if ligand exists, 0 if no ligand
df["Has_Ligand"] = df["Ligand ID"].notnull().astype(int)

X = df[feature_cols].astype(float)
y = df["Has_Ligand"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class weights for imbalance
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
print(f"Class weights: {class_weight_dict}")

# Build Functional API model 
inputs = Input(shape=(X_train_scaled.shape[1],))
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training 
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=2
)

# Evaluation
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
print("\nModel performance on test set:")
print(classification_report(y_test, y_pred))

# Save files
model.save(MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
print(f"Saved model -> {MODEL_OUT}")
print(f"Saved scaler -> {SCALER_OUT}")