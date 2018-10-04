
import numpy as np, pandas as pd, pickle, copy
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor


def bestLagRegr(dataMat, maxLag, minLag, yName):

    # Purpose: identify the best lag for variables to be used in regression. Uses CrossValidation
    # (folds TimesSeriesSplit)

    # data - a panda DF with named columns
    # maxLag - the maximum lag to be tested. Integer
    # minLag - the minimum lag to be tested. Integer
    # yName - the name of the column of 'data' containing the dependent var. String

    data = dataMat.copy()
    colnames = [y for y in data.columns if y != yName]
    lags = range(minLag, maxLag)
    folds = TimeSeriesSplit(n_splits=int(np.round(data.shape[0] / 10, 0)))

    results = {}

    for col in colnames:

        scores = []
        lags_list = []

        for l in lags:
            varname = col + str('_lag') + str(l)
            data[varname] = data[col].shift(l)
            YX = data[[yName, varname]]
            YX = YX.dropna().as_matrix()

            # Build regressor and estimate metric of prediction performance with CV
            regr = ensemble.GradientBoostingRegressor(learning_rate=0.01, max_depth=1, n_estimators=500)
            perform = cross_val_score(regr, X=YX[:, 1].reshape(-1, 1), y=YX[:, 0], cv=folds,
                                      scoring='neg_median_absolute_error')

            # Store scores result and lags
            scores.append(np.median(perform))
            lags_list.append(l)

        # Calculate best score, corresponding best lag and store it in a dictionary, containing all colnames
        best_score = max(scores)
        best_lag = lags_list[scores.index(best_score)]
        results[col] = [best_lag, best_score]

    return(results)


def laggedDataMat(dataMat, yName, lagsDict):

    # Purpose: build a lagged DF
    # dataMat: the unlagged DF with named columns
    # yName: name of the dependent var. String
    # lagsDict: dictionary produced with 'bestLagRegr, containing:
    #                                                               - keys: column names of dataMat
    #                                                               - elements: lists with lag, CV score
    #
    # Output: a panda DataFrame with columns order sorted alphabetically

    # Initialize empty DF
    df = pd.DataFrame(index=dataMat.index)
    # Set dependent var
    df[yName] = dataMat[[yName]]

    # Creating and adding the lagged vars
    for colName in lagsDict.keys():
        l = lagsDict[colName][0]
        colNameLag = colName + str('_lag') + str(l)
        df[colNameLag] = dataMat[[colName]].shift(l)

    df = df.sort_index(axis=1)

    return(df)


def modelSelection(maxLag, data, depVar, toSave, pSpace = None, alpha = 0.95):

    if pSpace is None:
        pSpace = dict(n_estimators=list(range(5, 2000, 10)),
                      learning_rate=list(np.arange(0.001, 1, 0.1)),
                      max_depth=list(range(1, 3, 1)))

    lags = range(1, maxLag)
    results = dict()

    for lagMin in lags:
        print('Esimating model for lag: ', lagMin)
        lagAnalysis = bestLagRegr(data, maxLag, lagMin, depVar)
        lagMat = laggedDataMat(data, depVar, lagAnalysis)
        lagMat = lagMat.dropna()
        trainY = np.ravel(lagMat[depVar])
        lagMat = lagMat.drop([depVar], 1)

        folds = TimeSeriesSplit(n_splits=int(round(lagMat.shape[0] / 10, 0)))

        model = GradientBoostingRegressor(loss='ls')
        regr = GridSearchCV(estimator=model, param_grid=pSpace, scoring='neg_mean_squared_error', cv=folds)
        regr.fit(lagMat, trainY)

        modelName = toSave + '/' + 'modelOI' + '_lag' + str(lagMin) + '.sav'
        pickle.dump(regr.best_estimator_, open(modelName, 'wb'))
        temp = dict(BestModel=regr.best_estimator_, Score=regr.best_score_, Lags=lagAnalysis)

        regrQUpper = copy.deepcopy(regr.best_estimator_)
        regrQUpper.set_params(loss='quantile', alpha=alpha)
        regrQUpper.fit(lagMat, trainY)
        temp['QUpper'] = regrQUpper

        regrQLower = copy.deepcopy(regr.best_estimator_)
        regrQLower.set_params(loss='quantile', alpha=(1-alpha))
        regrQLower.fit(lagMat, trainY)
        temp['QLower'] = regrQLower

        key = 'Lag ' + str(lagMin)
        results[key] = temp

    return(results)

