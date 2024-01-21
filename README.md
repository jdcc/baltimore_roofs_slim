There are three data sources:
  * A geodatabase (gdb) file
  * Tabular files (CSVs and Excel files)
  * Aerial images

These get merged into two data types:
  * Numerical features
  * Images

Images get passed into their own neural network model, which tries to predict whether the roof is extensively damaged. These predictions and the other numerical features get passed into a second model that makes the final predictions.