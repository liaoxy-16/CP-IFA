import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from minepy import MINE

# Load original unstandardized data for residual modeling
df = pd.read_csv('./data/sim1_test.csv')
print('df:\n', df)

columns = df.columns
mic_matrix = pd.DataFrame(0.0, index=columns, columns=columns)
print("MIC matrix:\n", mic_matrix)

mine = MINE(alpha=0.1, c=30)

for target_col in columns:
    for feature_col in columns:
        if target_col == feature_col:
            mic_matrix.loc[target_col, feature_col] = np.nan
            mic_matrix.loc[target_col, feature_col] = 0.0
        else:
            X = df[[feature_col]].values.reshape(-1, 1)
            y = df[target_col].values.reshape(-1, 1)

            mlp = MLPRegressor(hidden_layer_sizes=(20,20), max_iter=1000, random_state=42)

            mlp.fit(X, y.ravel())
            residuals = y - mlp.predict(X).reshape(-1, 1)

            mine.compute_score(X.ravel(), residuals.ravel())
            mic = mine.mic()

            mic_matrix.loc[target_col, feature_col] = mic
            print('target_col=',target_col,' feature_col=',feature_col,' MIC=',mic)


print("MIC matrix:\n", mic_matrix)
mic_matrix.to_csv('./result/Sim1_MIC_alpha=0.1,c=30_MLP2.csv')

comparison_matrix = pd.DataFrame(0,index=columns, columns=columns)

for i in range(len(columns)):
    for j in range(i+1, len(columns)):
        row = columns[i]
        col = columns[j]

        mic_ij = float(mic_matrix.loc[row, col])
        mic_ji = float(mic_matrix.loc[col, row])

        if mic_ij < mic_ji:
            comparison_matrix.loc[row, col] = 1
            comparison_matrix.loc[col, row] = 0
        else:
            comparison_matrix.loc[row, col] = 0
            comparison_matrix.loc[col, row] = 1

print("comparison_matrix:\n", comparison_matrix)

comparison_matrix.to_csv('./result/Causal_direction_Sim1_MIC_alpha=0.1,c=30_MLP2.csv')

