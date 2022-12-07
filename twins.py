import numpy as np
import pandas as pd

from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import KFold
from tqdm import tqdm
from learners import *

num_split = 10

t_all = pd.read_csv('./datasets/TWINS/twin_pairs_T_3years_samesex.csv')
x_all = pd.read_csv('./datasets/TWINS/twin_pairs_X_3years_samesex.csv')
y_all = pd.read_csv('./datasets/TWINS/twin_pairs_Y_3years_samesex.csv')

t_all = t_all.drop(t_all.columns[0], axis = 1)

x_all = x_all.drop(x_all.columns[:2],axis = 1)
x_all = x_all.drop(['infant_id_0', 'infant_id_1', 'bord_0', 'bord_1'], axis = 1)

y_all = y_all.drop(y_all.columns[0], axis = 1)

t_filter = t_all[(t_all<2000).all(axis=1)]
x_filter = x_all[(t_all<2000).all(axis=1)]
x_filter_na = x_filter.fillna(x_filter.mean())
y_filter = y_all[(t_all<2000).all(axis=1)]

num = len(t_filter)

treats_all = np.random.binomial(1, 0.5, num)
X_all = x_filter_na.values
outcomes_all = np.array([y_filter.values[i, treats_all[i]] for i in range(num)])

cates_all = np.array([y_filter.values[i, 1] - y_filter.values[i, 0] for i in range(num)])

kf = KFold(n_splits = num_split)
mse_s_cv = np.zeros(num_split)
mse_t_cv = np.zeros(num_split)
mse_r_cv = np.zeros(num_split)
mse_dr_cv = np.zeros(num_split)

s_r2 = np.zeros(num_split)
t_r2 = np.zeros(num_split)
r_r2 = np.zeros(num_split)
dr_r2 = np.zeros(num_split)

j = 0

for train_idxs, test_idxs in tqdm(kf.split(X_all)):

    xs = X_all[train_idxs]
    treats = treats_all[train_idxs]
    outcomes = outcomes_all[train_idxs]
    xs_test = X_all[test_idxs]

    cates_test = cates_all[test_idxs]

    mse_s_cv[j], s_r2[j] = s_learner(xs, treats, outcomes, xs_test, cates_test)
    mse_t_cv[j], t_r2[j]= t_learner(xs, treats, outcomes, xs_test, cates_test)
    mse_r_cv[j], r_r2[j] = r_learner(xs, treats, outcomes, xs_test, cates_test)
    mse_dr_cv[j], dr_r2[j] = dr_learner(xs, treats, outcomes, xs_test, cates_test)
    j+=1


print(np.mean(mse_s_cv), np.std(mse_s_cv)*1.96/np.sqrt(num_split))
print(np.mean(mse_t_cv), np.std(mse_t_cv)*1.96/np.sqrt(num_split))
print(np.mean(mse_r_cv), np.std(mse_r_cv)*1.96/np.sqrt(num_split))
print(np.mean(mse_dr_cv), np.std(mse_dr_cv)*1.96/np.sqrt(num_split))


print(np.mean(s_r2), np.std(s_r2)*1.96/np.sqrt(num_split))
print(np.mean(t_r2), np.std(t_r2)*1.96/np.sqrt(num_split))
print(np.mean(r_r2), np.std(r_r2)*1.96/np.sqrt(num_split))
print(np.mean(dr_r2), np.std(dr_r2)*1.96/np.sqrt(num_split))

# train_idxs = sample_without_replacement(num, num//2)
# n1 = len(train_idxs)
# test_idxs = np.setdiff1d(np.arange(num), train_idxs, True)
# n2 = len(test_idxs)

# xs, treats, outcomes = X_all[train_idxs], treats_all[train_idxs], outcomes_all[train_idxs]

# xs_test = X_all[test_idxs]

# cates_test = cates_all[test_idxs]