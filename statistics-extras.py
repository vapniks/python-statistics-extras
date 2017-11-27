# Miscellaneous helpful statistics tests and functions.

# Copyright 2017 Joe Bloggs (vapniks@yahoo.com)

import numpy as np
import math
import statistics as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

def vuong_test(model1,model2):
    '''Perform Vuong closeness test for comparing linear models. 
    model1 and model2 should be model.fit objects (from statsmodels) for non-nested 
    linear models with the same dependent variable, and estimated with the same dataset.
    model1 should be the model with higher likelihood.
    '''
    num_data = model1.nobs
    omegaw = stats.variance(model1.resid / model2.resid)
    numerator = (math.exp(model1.llf)-math.exp(model2.llf))-((model1.params.size-model2.params.size)/2)*math.log(num_data)
    test_stat = numerator/(omegaw*math.sqrt(num_data))
    pvalue = 1-stats.norm.cdf(test_stat)
    return {'pvalue':pvalue,'Z':test_stat}

def reset_ramsey(res, M=5):
    '''Ramsey's RESET specification test for linear models
    This is a general specification test, for additional non-linear effects
    in a model. res should be a linear model.fit object from statsmodels
    Notes
    -----
    The test fits an auxiliary OLS regression where the design matrix, exog,
    is augmented by powers 2 to degree of the fitted values. Then it performs
    an F-test whether these additional terms are significant.
    If the p-value of the f-test is below a threshold, e.g. 0.1, then this
    indicates that there might be additional non-linear effects in the model
    and that the linear model is mis-specified.
    References
    ----------
    http://en.wikipedia.org/wiki/Ramsey_RESET_test
    '''
    order = M + 1
    k_vars = res.model.exog.shape[1]
    #vander without constant and x:
    y_fitted_vander = np.vander(res.fittedvalues, order)[:, :-2] #drop constant
    exog = np.column_stack((res.model.exog, y_fitted_vander))
    res_aux = smf.OLS(res.model.endog, exog).fit()
    #r_matrix = np.eye(degree, exog.shape[1], k_vars)
    r_matrix = np.eye(M-1, exog.shape[1], k_vars)
    #df1 = degree - 1
    #df2 = exog.shape[0] - degree - res.df_model (without constant)
    return res_aux.f_test(r_matrix) #, r_matrix, res_aux

def modelstable1(modelsdict,coeffs=None,stats=None,dp=3):
    '''Create table (pandas dataframe) for comparing regression model coefficients.
    modelsdict should be a dictionary whose keys are the model names, and whose values
    are fitted model objects (from statsmodels). The resulting table will have one column 
    for each model, labelled by the keys of modelsdict.
    coeffs can be a list of coefficient names to include in the table. The default is to
    include all coefficients from all models.
    stats can be a dictionary whose keys are row labels and whose values are the names of
    model object members that can be evaluated to return a single statistic,
    e.g: stats = {"AIC":"aic","BIC":"bic"} will cause model.aic and model.bic to be eval'ed
    for each model and the statistics reported in the table.
    dp is the number of decimal places to include for the numbers in the table.
    '''
    defaultstats = {'AIC':'aic','R-squared':'rsquared','Adj R-squared':'rsquared_adj',
                        'F-stat':'fvalue[0][0]','F-stat p-value':'f_pvalue'}
    allcoeffs = list(set.union(*[set(v.params.index.values.tolist()) for v in modelsdict.values()]))
    allcoeffs.sort()
    if not coeffs:
        if 'Intercept' in allcoeffs:
            coeffs = ['Intercept']+[n for n in allcoeffs if n!='Intercept']
        else:
            coeffs = allcoeffs
            stats = stats if stats else defaultstats
            data = pd.DataFrame(index=coeffs+list(stats.keys()),columns=list(modelsdict.keys()))
            for k in modelsdict.keys():
                model = modelsdict[k]
                for coeff in data.index:
                    if coeff in model.params.index:
                        param = round(model.params[coeff],dp)
                        pval = model.pvalues[coeff]
                        data.loc[coeff,k] = str(param)+("***" if (pval < 0.001) else 
                                                        ("**" if (pval < 0.01) else 
                                                             ("*" if (pval < 0.05) else "")))
                    else:
                        data.loc[coeff,k] = ''
                        for stat in stats.keys():
                            data.loc[stat,k] = np.round(eval('model.'+stats[stat]),dp)
                            return(data)

def modelstable2(specsdict,data,coeffs=None,stats=None,cov_type='HC3',dp=3):
    '''Create table for comparing regression model coefficients given model specification strings.
    specsdict should be a dictionary whose keys are the column headers, and whose values are the
    corresponding model specifications. data is a data frame with which to evaluate the models.
    coeffs, stats, and dp are the same as for modelstable1.
    '''
    modelsdict = {}
    for specname in specsdict.keys():
        modelsdict[specname] = smf.ols(specsdict[specname],data).fit(cov_type=cov_type)
        return(modelstable1(modelsdict,coeffs=coeffs,stats=stats,dp=dp))
