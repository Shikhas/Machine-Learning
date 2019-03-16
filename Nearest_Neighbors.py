import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn

#reading data from Fraud csv and setting case-id as index
fraud_data = pandas.read_csv("Data/Fraud.csv").set_index("CASE_ID")


print("The percentge of investigations fraudulent are: ",fraud_data['FRAUD'][fraud_data['FRAUD'] == 1].count()/5960)


def boxplots_interval_vars(int_var):
    fraud_data.boxplot(column=int_var, by="FRAUD", vert=False)
    plt.title('')
    plt.suptitle("")
    plt.xlabel(int_var)
    plt.ylabel("Fraud")
    plt.show()

boxplots_interval_vars("TOTAL_SPEND")
boxplots_interval_vars("DOCTOR_VISITS")
boxplots_interval_vars("NUM_CLAIMS")
boxplots_interval_vars("MEMBER_DURATION")
boxplots_interval_vars("OPTOM_PRESC")
boxplots_interval_vars("NUM_MEMBERS")


int_vars_list = ["TOTAL_SPEND","DOCTOR_VISITS","NUM_CLAIMS","MEMBER_DURATION","OPTOM_PRESC","NUM_MEMBERS"]

def orthonorm_int_var(int_var):
    pass
x = numpy.matrix(fraud_data[int_vars_list])
xtx = x.transpose()*x
evals, evecs = numpy.linalg.eigh(xtx)
print("Eigenvalues: ", evals) #did not filter as all the values are greater than 1 so #dim=6
transf = evecs*numpy.linalg.inv(numpy.sqrt(numpy.diagflat(evals)))
transf_x = x*transf
print("The transformation matrix is \n",transf)
xtx = transf_x.transpose()*transf_x
print("Expect an identity matrix = \n", xtx)


KNNSpec = knn(n_neighbors = 5, algorithm = 'brute')
fitted_model = KNNSpec.fit(transf_x, fraud_data["FRAUD"])
score = fitted_model.score(transf_x,fraud_data["FRAUD"])
print("The score of the fitted model is:",score)


new_obs = [[7500,15,3,127,2,2]] #list of records and each record is a list

trans_new_obs = new_obs * transf
neighbor_new_obs= fitted_model.kneighbors(trans_new_obs, return_distance=False)
print("The neighbors of the new observation is: ",neighbor_new_obs)
for i in neighbor_new_obs.tolist()[0]:
    print("index", i)
    print(transf_x[i],fraud_data["FRAUD"][i])


print("The predicted value for the fitted model is:",fitted_model.predict(trans_new_obs))
