import matplotlib.pyplot as plt
import numpy
import pandas
import statsmodels.api as stats

PurchaseLikelihood_data = pandas.read_csv("Data/Purchase_Likelihood.csv"
                                      )
#print(PurchaseLikelihood_data)

#print(" There are %(params)d parameters in model with only Intercet term." %{ "params": 2}) # k-1 = 2, k are number
# of values for target variable

crossTable = pandas.crosstab(index=PurchaseLikelihood_data['A'], columns=["Count"], margins=True, dropna=False).\
    drop(columns=["All"])
#import ipdb; ipdb.set_trace()
print(crossTable)

A0_count = crossTable.loc[0]["Count"]
A1_count = crossTable.loc[1]["Count"]
A2_count = crossTable.loc[2]["Count"]
total_count = crossTable.loc["All"]["Count"]

def mle_estimate_without_MNlogit(A0_count,A1_count,A2_count):
    #pi_ij = n_ij/ni
    p_0 = A0_count/total_count
    p_1 = A1_count/total_count
    p_2 = A2_count/total_count
    return(p_0,p_1,p_2)

def log_likelihood_without_MNlogit(A0_count,A1_count,A2_count):
    l = 0
    (p0,p1,p2) = mle_estimate_without_MNlogit(A0_count,A1_count,A2_count)
    l = A0_count*numpy.log(p0) + A1_count*numpy.log(p1) + A2_count*numpy.log(p2)
    return(l)

print("The estimates for J=1,2,3 without calling MNlogit function are respectively: ",mle_estimate_without_MNlogit(A0_count,A1_count,A2_count))
print("The log-likelihood of intercept-only model is:",log_likelihood_without_MNlogit(A0_count,A1_count,A2_count))

(p_0,p_1,p_2) = mle_estimate_without_MNlogit(A0_count,A1_count,A2_count)
p_1J = 1 # as in intercept-only model we have no predictors so i value is 1
beta_10 = 0 # as given in question we need to take beta_10=0, so that  as reference will be p_0
beta_20 = numpy.log(p_1/p_0)
beta_30 = numpy.log(p_2/p_0)
print("The maximum likelihood estimates of Intercept terms for j=1 is %(b_10)d ,j=2 is %(b_20)f ,j=3 is %(b_30)f"
      %{"b_10": beta_10, "b_20": beta_20, "b_30": beta_30})

# creating contingency table of frequecy and percentage with where group_size, homeowner, and married_couple are on the
# row dimension, and A is on the column dimension
pur_likelihood = PurchaseLikelihood_data['A'].values # creating 1-d numpy array using .values
#pur_likelihood = PurchaseLikelihood_data[['A']].values # this is not to be used as creates 2-d array
g_size = PurchaseLikelihood_data['group_size'].values
home_owner= PurchaseLikelihood_data['homeowner'].values
married_couple = PurchaseLikelihood_data['married_couple'].values
countTable = pandas.crosstab([g_size, home_owner,married_couple], pur_likelihood, rownames=['group_size', 'home_owner'
    ,'married_couple'], colnames=['purchase_likelihood'])
percentTable = countTable.div(countTable.sum(1), axis='index')*100
print("Frequency Table: \n")
print(countTable)
print( )
print("Percent Table: \n")
print(percentTable)


purchase_likelihood = PurchaseLikelihood_data['A'].astype('category') #astype will convert pandas object to
# categorical type
y = purchase_likelihood
y_category = y.cat.categories

grp_size = PurchaseLikelihood_data[['group_size']].astype('category')
X = pandas.get_dummies(grp_size)
x2 = pandas.get_dummies(PurchaseLikelihood_data[['homeowner']].astype('category'))
x3 = pandas.get_dummies(PurchaseLikelihood_data[['married_couple']].astype('category'))
X = X.join([x2,x3]) # passing a list of dataframes
X = stats.add_constant(X, prepend=True)
#import ipdb;ipdb.set_trace()
logit = stats.MNLogit(y, X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))


logit_category_1 = X.dot(thisParameter.iloc[:, 0])
logit_category_2 = X.dot(thisParameter.iloc[:, 1])
prob_category_1_ref_0 = numpy.exp(logit_category_1)
print(f"Values of group_size, homeowner, married_couple for which odd Prob(A=1)/Prob(A = 0) will attain its maximum"
      f" value of {prob_category_1_ref_0.max()}:\n")
print(X.iloc[prob_category_1_ref_0.idxmax(), 1:])  # part(i)

print(f"Odds ratio for group size=3 versus group size=1, and A=2 versus A=0 is: "
      f"{numpy.exp(thisParameter.loc['group_size_3'][1] - thisParameter.loc['group_size_1'][1])}")  # part(j)

params_category_2_to_category_1 = thisParameter.iloc[:,1] - thisParameter.iloc[:,0]
print(f"Odds ratio for group size=1 versus group size=3, and A=2 versus A=1 is: "
      f"{numpy.exp(params_category_2_to_category_1.loc['group_size_1'] - params_category_2_to_category_1.loc['group_size_3'])}")
# part(k)