import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.utils.random import sample_without_replacement


def s_learner(xs_train, treats_train, outcomes_train, xs_test, cates_test):
    n_train = len(outcomes_train)
    n_test = len(xs_test)

    if len(np.shape(xs_train))==1:
        X = np.concatenate((np.expand_dims(xs_train, axis = 1), np.expand_dims(treats_train, axis = 1)), axis = 1)
    else:
        X = np.concatenate((xs_train, np.expand_dims(treats_train, axis = 1)), axis = 1)
    y = outcomes_train
    model = make_pipeline(SplineTransformer(), LinearRegression())
    model.fit(X,y)

    if len(np.shape(xs_test))==1:
        X_test = np.expand_dims(xs_test, axis = 1)
    else:
        X_test = xs_test

    
    CATEs = np.zeros(n_test)
    for i in range(n_test):
        treated = model.predict([np.concatenate((X_test[i], np.array([1])))])[0]
        control = model.predict([np.concatenate((X_test[i], np.array([0])))])[0]
        CATEs[i] = treated - control
    
    return mean_squared_error(CATEs, cates_test)


def t_learner(xs_train, treats_train, outcomes_train, xs_test, cates_test):
    n_train = len(outcomes_train)
    n_test = len(xs_test)

    if len(np.shape(xs_train))==1:
        X = np.expand_dims(xs_train, axis = 1)
    else:
        X = xs_train
    y = outcomes_train
    model1 = make_pipeline(SplineTransformer(), LinearRegression())
    model1.fit(X[treats_train == 1],y[treats_train==1])
    model0 = make_pipeline(SplineTransformer(), LinearRegression())
    model0.fit(X[treats_train == 0],y[treats_train==0])

    if len(np.shape(xs_test))==1:
        X_test = np.expand_dims(xs_test, axis = 1)
    else:
        X_test = xs_test
    
    CATEs = np.zeros(n_test)
    for i in range(n_test):
        treated = model1.predict([X_test[i]])[0]
        control = model0.predict([X_test[i]])[0]
        CATEs[i] = treated - control
    
    return mean_squared_error(CATEs, cates_test)

def r_learner(xs_train, treats_train, outcomes_train, xs_test, cates_test):
    n_train = len(outcomes_train)
    n_test = len(xs_test)

    if len(np.shape(xs_train))==1:
        X = np.expand_dims(xs_train, axis = 1)
    else:
        X = xs_train
    y = outcomes_train

    half1_idxs_train = sample_without_replacement(n_train, n_train//2)
    n1 = len(half1_idxs_train)
    half2_idxs_train = np.setdiff1d(np.arange(n_train), half1_idxs_train, True)
    n2 = len(half2_idxs_train)

    X1, treats_train1, y1 = X[half1_idxs_train], treats_train[half1_idxs_train], y[half1_idxs_train]
    X2, treats_train2, y2 = X[half2_idxs_train], treats_train[half2_idxs_train], y[half2_idxs_train]

    model1 = make_pipeline(SplineTransformer(), LinearRegression())
    model1.fit(X1,y1)

    model1p = make_pipeline(SplineTransformer(), LogisticRegression())
    model1p.fit(X1,treats_train1)

    model2 = make_pipeline(SplineTransformer(), LinearRegression())
    model2.fit(X2, y2)

    model2p = make_pipeline(SplineTransformer(), LogisticRegression())
    model2p.fit(X2, treats_train2)

    y_residuals = np.zeros(n_train)

    e_residuals = np.zeros(n_train)

    half1_idxs_set = set(half1_idxs_train) #for ~O(1) membership checking
    half2_idxs_set = set(half2_idxs_train)

    for i in range(n_train):
        if i in half1_idxs_set:
            y_residuals[i] = y[i] - model2.predict([X[i]])
            e_residuals[i] = treats_train[i] - model2p.predict_proba([X[i]])[0][1]
        else:
            y_residuals[i] = y[i] - model1.predict([X[i]])
            e_residuals[i] = treats_train[i] - model1p.predict_proba([X[i]])[0][1]
    
    pseudos = y_residuals/e_residuals
    weights = np.power(e_residuals, 2)

    model3 = make_pipeline(SplineTransformer(), LinearRegression())
    model3.fit(X, pseudos, **{'linearregression__sample_weight':weights})

    if len(np.shape(xs_test))==1:
        X_test = np.expand_dims(xs_test, axis = 1)
    else:
        X_test = xs_test

    CATEs = model3.predict(X_test)

    return mean_squared_error(CATEs, cates_test)


def dr_learner(xs_train, treats_train, outcomes_train, xs_test, cates_test):
    n_train = len(outcomes_train)
    n_test = len(xs_test)

    if len(np.shape(xs_train))==1:
        X = np.expand_dims(xs_train, axis = 1)
    else:
        X = xs_train
    y = outcomes_train

    stage1_idxs_train = sample_without_replacement(n_train, n_train//2)
    n1 = len(stage1_idxs_train)
    stage2_idxs_train = np.setdiff1d(np.arange(n_train), stage1_idxs_train, True)
    n2 = len(stage2_idxs_train)

    X1, treats_train1, y1 = X[stage1_idxs_train], treats_train[stage1_idxs_train], y[stage1_idxs_train]
    X2, treats_train2, y2 = X[stage2_idxs_train], treats_train[stage2_idxs_train], y[stage2_idxs_train]


    model1 = make_pipeline(SplineTransformer(), LinearRegression())
    model1.fit(X1[treats_train1 == 1],y1[treats_train1==1])
    model0 = make_pipeline(SplineTransformer(), LinearRegression())
    model0.fit(X1[treats_train1 == 0],y1[treats_train1==0])
    
    modelp = make_pipeline(SplineTransformer(), LogisticRegression())
    modelp.fit(X1, treats_train1)


    pseudos = np.zeros(n2)

    for i in range(n2):
        pseudos[i] = model1.predict([X2[i]])[0] - model0.predict([X2[i]])[0]
        if treats_train2[i]:
            pseudos[i]+= (1/modelp.predict_proba([X2[i]])[0][1])*(y2[i] - model1.predict([X2[i]])[0])
        else:
            pseudos[i]-= (1/modelp.predict_proba([X2[i]])[0][0])*(y2[i] - model0.predict([X2[i]])[0])

    model2 = make_pipeline(SplineTransformer(), LinearRegression())
    model2.fit(X2,pseudos)

    if len(np.shape(xs_test))==1:
        X_test = np.expand_dims(xs_test, axis = 1)
    else:
        X_test = xs_test

    CATEs = model2.predict(X_test)

    return mean_squared_error(CATEs, cates_test)