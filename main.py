import sys
import os

# Add src folder manually to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.append(src_path)

from src.data_preprocessing import load_and_clean
from src.model_training import train_model
from src.model_evaluation import evaluate
df = load_and_clean(r"C:\Users\fardh\OneDrive\Desktop\numpy\customer-churn-prediction\customer_churn.csv")

model, X_test, y_test = train_model(df)

results = evaluate(model, X_test, y_test)

print("\nModel Performance:")
print(results)