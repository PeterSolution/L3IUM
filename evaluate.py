
import os

from dotenv import load_dotenv
import joblib
from sklearn.metrics import accuracy_score

from data_prep import load_data, prepare_data






load_dotenv()
modeldirectory=os.getenv("MODEL_DIR", "models")

model=joblib.load(os.path.join(modeldirectory, os.getenv("MODEL_FILENAME", "model.pkl")))
df=load_data(os.getenv("DATA_PATH", "data/wine.csv"))
x_train, x_test, y_train, y_test=prepare_data(df)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")

with open(os.path.join(modeldirectory, os.getenv("RESULTS_PATH", "results.txt")), 'w') as f:
    f.write(f"Model Accuracy: {accuracy}\n")