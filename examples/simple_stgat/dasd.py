
import pandas as pd
import numpy as np



df = pd.DataFrame([[np.nan, 2, 6, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5],
                   [np.nan, 3, np.nan, 4]],
                  columns=list('ABCD'))

df = df.fillna(df.mean())
print(type(df))
print(df)
