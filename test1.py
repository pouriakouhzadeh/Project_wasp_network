from scipy.signal import lfilter
import pandas as pd
import os
import numpy as np

INITIAL_CSV_DIRECTORY = "/home/pouria/project/csv_files_initial/"
NEW_CSV_DIRECTORY = "/home/pouria/project/csv_files/"

def json_to_df(json_data):
    return pd.read_json(json_data, orient='split')

def df_to_json(df):
    # Setting double precision to 15 to maintain floating point accuracy
    return df.to_json(orient='split', double_precision=15)

def read_data(file_path, tail_size=21000):
    try:
        data = pd.read_csv(file_path)
        data = data.tail(tail_size)
        data.reset_index(inplace=True, drop=True)
        return data
    except FileNotFoundError:
        return None

data = read_data(f"{os.path.join(NEW_CSV_DIRECTORY, 'EURUSD60' + '.csv')}")
N = 5  # طول فیلتر
b = np.ones(N) / N
a = 1  # فیلتر FIR، پس a = 1
data['Filtered'] = lfilter(b, a, data['close'])
df1 = df_to_json(data)
df2 = json_to_df(df1)

# Using np.allclose to compare floating point numbers
for i in range(len(data)):
    for j in range(len(data.columns)):
        if isinstance(data.iloc[i, j], (float, np.floating)):
            if not np.allclose(data.iloc[i, j], df2.iloc[i, j], atol=1e-8):
                print(f"ERROR at row {i}, column {j}: {data.iloc[i, j]} != {df2.iloc[i, j]}")
        else:
            if data.iloc[i, j] != df2.iloc[i, j]:
                print(f"ERROR at row {i}, column {j}: {data.iloc[i, j]} != {df2.iloc[i, j]}")

print(data)
