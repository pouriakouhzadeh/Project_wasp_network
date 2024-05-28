from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

class FeatureSelection:
    def select(self, data, target, n, indices):
        data = pd.DataFrame(data)
        selected_indices = indices[:n]
        selected_features = data.columns[selected_indices]
        X_selected = data[selected_features]
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features

# Example usage:
# Assuming `data` is a DataFrame and `target` is the target column
# fs = FeatureSelection()
# X_selected, selected_features = fs.select(data, target, n=5, n_estimators=100, max_depth=5)
# print("Selected features:", selected_features)
