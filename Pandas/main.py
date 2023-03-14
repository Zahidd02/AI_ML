# Data Cleaning - Exercise
import numpy as np
import pandas as pd

# 'header' is set to 'None' because we will replace the column name, or else it will replace first row as column.
df = pd.read_excel('bridge.xlsx', header=None)
df.columns = ['IDENTIF', 'RIVER', 'LOCATION', 'ERECTED', 'PURPOSE', 'LENGTH', 'LANES', 'CLEAR-G',
              'T-OR-D', 'MATERIAL', 'SPAN', 'REL-L', 'TYPE']  # column rename

question_mark_count = {}  # Stores column name + number of "?" in that column
for x in range(0, len(df.iloc[0, :])):
    name = df.columns[x]
    if "?" in df[name].value_counts():
        question_mark_count[name] = df[name].value_counts()["?"]
    else:
        question_mark_count[name] = 0

for y in df.columns:
    count = 0
    if y.endswith(('N', 'H', 'S')):
        temp_df = pd.DataFrame(df[y].value_counts())
        for z in range(0, len(temp_df)):
            if temp_df.index[z] != "?":
                # temp_df.index[z] fetches the index of current iteration and '0' is the value in first column
                count += temp_df.loc[temp_df.index[z]][0]
        if count < 100:
            df.drop([y], axis=1, inplace=True)
        del temp_df

df.replace("?", np.NaN, inplace=True)
df.dropna(thresh=8, inplace=True)  # Keep the row with at least 8 non-NA value.

for name in df.columns:
    column_mode = df[name].mode()[0]
    df[name].replace(np.nan, column_mode, inplace=True)

with pd.ExcelWriter('Final_xl.xlsx') as writer:
    df.to_excel(writer)