# Intership-project
3D Printer Material Prediction Using Machine Learning
# 1. Install Required Libraries (if needed)
!pip install -q scikit-learn joblib

# 2. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 3. Upload Dataset from Local Machine
from google.colab import files
uploaded = files.upload()

# 4. Load Dataset
import io
df = pd.read_csv(io.BytesIO(uploaded['3d_printer_material_dataset.csv']))  # Use the exact file name

# 5. Encode Target Variable
le = LabelEncoder()
df['Material'] = le.fit_transform(df['Material'])

# 6. Split Features and Labels
X = df.drop('Material', axis=1)
y = df['Material']

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 9. Predictions and Evaluation
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 10. Save Model and Label Encoder
joblib.dump(model, "3d_material_model.joblib")
joblib.dump(le, "label_encoder.joblib")

# 11. Download Model Files
files.download("3d_material_model.joblib")
files.download("label_encoder.joblib")
