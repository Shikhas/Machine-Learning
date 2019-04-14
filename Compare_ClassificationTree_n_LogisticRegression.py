import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import math
from sklearn.model_selection import train_test_split
import statsmodels.api as stats
import matplotlib.pyplot as plt

policy_holder_data = pandas.read_csv("Data/policy_2001.csv",)
policy_holder_data = policy_holder_data[["CLAIM_FLAG","CREDIT_SCORE_BAND","BLUEBOOK_1000","CUST_LOYALTY","MVR_PTS","TIF","TRAVTIME"]]

print(policy_holder_data.columns)
# We perform startified simple random sampling to maintain distribution of target variable in training and testing data
print('Number of Observations = ', policy_holder_data.shape[0])
print(policy_holder_data.groupby('CLAIM_FLAG').size() / policy_holder_data.shape[0])

#### dummy values for nominal
policy_holder_data = policy_holder_data.join(pandas.get_dummies(
    policy_holder_data[['CREDIT_SCORE_BAND']].astype('category')))
policy_holder_data = policy_holder_data.drop("CREDIT_SCORE_BAND", axis=1)
policy_holder_data["CLAIM_FLAG"] = policy_holder_data[['CLAIM_FLAG']].astype('category')


policy_holder_train, policy_holder_test = train_test_split(policy_holder_data, test_size = 0.25, random_state =
20190402, stratify = policy_holder_data['CLAIM_FLAG'])
print('Number of Observations in Training = ', policy_holder_train.shape[0])
print('Number of Observations in Testing = ', policy_holder_test.shape[0])
print(policy_holder_train.groupby('CLAIM_FLAG').size() / policy_holder_train.shape[0])
print(policy_holder_test.groupby('CLAIM_FLAG').size() / policy_holder_test.shape[0])



#The distribution of CLAIM_FLAG in each partition is very similar to that of the original data.

# Model will be build using training data and compared by calculating metrics over testing data

########### Classification Tree Model  ##################
#import ipdb;ipdb.set_trace()
X_train = (policy_holder_train.drop(columns = ["CLAIM_FLAG"]))
Y_train = policy_holder_train["CLAIM_FLAG"]

X_test = (policy_holder_test.drop(columns = ["CLAIM_FLAG"]))
Y_test = policy_holder_test["CLAIM_FLAG"]


from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20190402)

claim_DT = classTree.fit(X_train, Y_train)
claim_nonclaim_DT_prob = classTree.predict_proba(X_test)
claim_DT_prob = claim_nonclaim_DT_prob[:, 1]
DT_pred = classTree.predict(X_test)


print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(classTree.score(X_train, Y_train)))
print('Accuracy of Decision Tree classifier on testing set: {:.6f}' .format(classTree.score(X_test, Y_test)))
import graphviz
dot_data = tree.export_graphviz(claim_DT,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = X_train.columns,
                                class_names = ['0', '1'])

graph = graphviz.Source(dot_data)
print(graph)

graph.render('mid_test_12',".")

########### Logit Model  ##################



X_test = stats.add_constant(X_test, prepend=True)
X_train = stats.add_constant(X_train, prepend=True)

#import ipdb;ipdb.set_trace()
logit = stats.Logit(Y_train, X_train )
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
print("Model Parameter Estimates:\n", thisFit.params)
#print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

n_X_test = X_test.shape[0]
X_b = X_test.dot(thisParameter)
prob_non_claim = 1/(1 + numpy.exp(X_b))
prob_claim = 1 - prob_non_claim

mn_pred = numpy.zeros(n_X_test)
mn_pred[prob_claim >= prob_non_claim] = 1

##################### Evaluation Metrics ############################


def evalution_metrics(predProbY,nY, Y, pred_by_model):
    # Determine the predicted class of Y
    true_y_val = Y.values
    predY = numpy.zeros(nY)
    for i in range(nY):
        #if predProbY[i] >= 0.287879:
        if predProbY[i] >= 0.2879:
            predY[i] = 1
        else:
            predY[i] = 0
    # Calculate the Root Average Squared Error

    RASE = numpy.sqrt(metrics.mean_squared_error(true_y_val, predProbY))
    # Calculate the Root Mean Squared Error
    # Y_true = 1.0 * numpy.isin(Y, 1)
    '''
    RMSE = metrics.mean_squared_error(true_y_val, predProbY)
    RMSE = numpy.sqrt(RMSE)
    '''
    # For binary y_true, y_score is supposed to be the score of the class with greater label.
    AUC = metrics.roc_auc_score(true_y_val, pred_by_model)
    accuracy = metrics.accuracy_score(Y, predY)

    print('                  Accuracy: {:.13f}' .format(accuracy))
    print('    Misclassification Rate: {:.13f}' .format(1-accuracy))
    print('          Area Under Curve: {:.13f}' .format(AUC))
    print('Root Average Squared Error: {:.13f}' .format(RASE))
    # print('   Root Mean Squared Error: {:.13f}' .format(RMSE))

##################### ROC curve generator  ############################


def roc_curve_generator(Y, predProbY, pred_prob_l):
    # Generate the coordinates for the ROC curve

    plt.figure()
    axs = plt.gca()

    fp, tp, _ = metrics.roc_curve(Y, predProbY, pos_label=1)
    axs.plot(fp,tp, color='blue', marker='o',  label="Decision Tree")
    fp, tp, _ = metrics.roc_curve(Y, pred_prob_l, pos_label=1)
    axs.plot(fp,tp, color='green',marker='o', label ='Logistic Regression')
    axs.plot([0, 1], [0, 1], color='red', linestyle=':',label='Uniformly Random Model')
    plt.legend()
    plt.grid(True)
    plt.xlabel("1 - Specificity (False Positive Rate)")
    plt.ylabel("Sensitivity (True Positive Rate)")
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()




print("Metrics for Decision Tree")
evalution_metrics(claim_DT_prob,X_test.shape[0],Y_test, DT_pred)
print("Metrics for logistic regression")
evalution_metrics(prob_claim.values, X_test.shape[0],Y_test, mn_pred)

roc_curve_generator(Y_test, claim_DT_prob, prob_claim.values)
