import numpy as np
import matplotlib.pyplot as plt
from lib import utils
from lib.metrics import masked_rmse_np, masked_mape_np, masked_mae_np
from lib.utils import StandardScaler
import pandas as pd

data = np.load('data/dcrnn_predictions.npz')
print(data.files)

prediction=data['prediction']
y_predict =prediction[0,:,:]
print(y_predict.shape)
y_predict=pd.DataFrame(y_predict)

truth=data['truth']
y_test=truth[0,:,:]
print(y_test.shape)
y_test=pd.DataFrame(y_test)
print(y_predict.values)
rmse = masked_rmse_np(preds=y_predict.values, labels=y_test.values, null_val=0)
mape = masked_mape_np(preds=y_predict.values, labels=y_test.values, null_val=0)
mae = masked_mae_np(preds=y_predict.values, labels=y_test.values, null_val=0)
print(rmse,mape* 100,mae)