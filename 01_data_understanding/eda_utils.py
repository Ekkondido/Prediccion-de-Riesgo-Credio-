import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.ticker as ticker


# Cargar los datos
df_app = pd.read_parquet("Data/Raw/application_parquet")
df_bureau = pd.read_parquet("Data/Raw/bureau_parquet")
df_prev_app = pd.read_parquet("Data/Raw/previous_application_parquet")

# Visualizar primeras filas
df_app.head()
