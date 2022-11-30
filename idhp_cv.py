import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold

from learners import *

np.random.seed(6746)

num_split = 10

ihdp = pd.read_csv('./datasets/IHDP/csv/ihdp_npci_1.csv', header = None)

kf = KFold(n_splits = num_split)

mse_s_cv = np.zeros(num_split)
mse_t_cv = np.zeros(num_split)
mse_r_cv = np.zeros(num_split)
mse_dr_cv = np.zeros(num_split)

j = 0

for train_idxs, test_idxs in tqdm(kf.split(ihdp.values)):
    ihdp_train = ihdp.iloc[train_idxs]
    ihdp_test = ihdp.iloc[test_idxs]

    xs = ihdp_train.iloc[:,-25:].values
    treats = ihdp_train.iloc[:,0].values
    outcomes = ihdp_train.iloc[:,1].values
    xs_test = ihdp_test.iloc[:,-25:].values

    cates_test = ihdp_test.iloc[:,4]-ihdp_test.iloc[:,3]

    mse_s_cv[j] = s_learner(xs, treats, outcomes, xs_test, cates_test)
    mse_t_cv[j] = t_learner(xs, treats, outcomes, xs_test, cates_test)
    mse_r_cv[j] = r_learner(xs, treats, outcomes, xs_test, cates_test)
    mse_dr_cv[j] = dr_learner(xs, treats, outcomes, xs_test, cates_test)
    j+=1


print(np.mean(mse_s_cv), np.std(mse_s_cv)*1.96/np.sqrt(num_split))
print(np.mean(mse_t_cv), np.std(mse_t_cv)*1.96/np.sqrt(num_split))
print(np.mean(mse_r_cv), np.std(mse_r_cv)*1.96/np.sqrt(num_split))
print(np.mean(mse_dr_cv), np.std(mse_dr_cv)*1.96/np.sqrt(num_split))




