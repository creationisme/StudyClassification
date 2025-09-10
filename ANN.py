import numpy as np
import pandas as pd
import torch


df = pd.read_csv("NNDataset.csv")
X = df[df.columns[:-1]]
y = df["quality"]
print(X.columns)