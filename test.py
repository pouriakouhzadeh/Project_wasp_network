# from sklearn.metrics import f1_score

# # نمونه داده‌های واقعی و پیش‌بینی شده
# y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
# y_pred = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]

# # محاسبه F1 Score
# f1 = f1_score(y_true, y_pred)

# print(f'F1 Score: {f1:.2f}')





























import pandas as pd
import pickle

def read_data(file_name, tail_size=25000):
    file_path = f"/home/pouria/project/csv_files_initial/{file_name}"
    try:
        data = pd.read_csv(file_path)
        data = data.tail(tail_size)
        data.reset_index(inplace=True, drop=True)
        return data
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None


currency_pair = "EURGBP60"
currency_data_for_save =  read_data(f"{currency_pair}.csv", tail_size= 25000)
print((currency_data_for_save))


# باز کردن فایل مدل برای خواندن
with open(f"/home/pouria/project/trained_models/{currency_pair}_features_indicts.pkl", 'rb') as f:
    # لود کردن مدل از فایل
    loaded_model = pickle.load(f)

# استفاده از مدل لود شده
print(loaded_model)