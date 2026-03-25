import pandas as pd

# 读取CSV文件为DataFrame
df = pd.read_csv('./result/all_losses_epoch=100.csv',index_col=0)
print('df:\n',df)

# Extract the 'Original' loss column
original_col = df['Original']
print('original_col:\n',original_col)

# Calculate the difference: subtract the 'Original' loss from the perturbed losses
# This difference represents the Causal Sensitivity Score
new_df = df.apply(lambda x: x - original_col if x.name != 'Original' else x)
print('new_df :\n',new_df)


new_df = new_df.where(pd.notna(df), df)
print('new_df :\n',new_df)

new_df.to_csv('./result/all_losses_epoch=100_difference.csv')

