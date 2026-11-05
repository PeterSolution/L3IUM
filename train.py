from dotenv import load_dotenv
import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_prep import load_data, prepare_data

load_dotenv()
modeldirectory=os.getenv("MODEL_DIR", "models")
modelFilename=os.getenv("MODEL_FILENAME", "model.pkl")

df=load_data(file_path=os.getenv("DATA_PATH", "data/wine.csv"))
x_train, x_test, y_train, y_test=prepare_data(df)

model=Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        max_iter=int(os.getenv("MAX_ITER", 1000)),
        C=float(os.getenv("C", 1.0)),
        random_state=int(os.getenv("RANDOM_SEED", 42))
        ))
])

model.fit(x_train, y_train)

os.makedirs(modeldirectory, exist_ok=True)
with open(os.path.join(modeldirectory, modelFilename), 'wb') as f:
    joblib.dump(model, f)
    