import os
from sklearn.datasets import load_wine

os.makedirs("data", exist_ok=True)
df = load_wine(as_frame=True).frame
df.to_csv("data/wine.csv", index=False)
print(df.shape)