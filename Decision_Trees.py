import matplotlib.pyplot as plt
import numpy
import pandas
import itertools

CustomerSurvey_data = pandas.read_csv("Data/CustomerSurveyData.csv",
                                      index_col="CustomerID")[['CreditCard', 'JobCategory', 'CarOwnership']]

# Check which predictors contain missing values
print(CustomerSurvey_data['CreditCard'].unique())
# print(CustomerSurvey_data['JobCategory'].unique())
# print(CustomerSurvey_data['CarOwnership'].unique())

# we can observe missing values in car ownership and job category
# CustomerSurvey_data = CustomerSurvey_data[CustomerSurvey_data["CarOwnership"] != "None"]
# print(CustomerSurvey_data['CarOwnership'].unique()) # check that the missing values are removed.

# if CustomerSurvey_data[CustomerSurvey_data["JobCategory"] == "nan"] :
#     CustomerSurvey_data = CustomerSurvey_data[CustomerSurvey_data["JobCategory"] = "Missing"]

CustomerSurvey_data.loc[CustomerSurvey_data["JobCategory"].isna(), "JobCategory"] = "Missing"
# print(CustomerSurvey_data['JobCategory'].unique())

# To get the counts of all the target variable by building contingency table
crossTable = pandas.crosstab(index=CustomerSurvey_data['CarOwnership'], columns=["Count"], margins=True, dropna=False)
crossTable = crossTable.drop(columns=['All'])
# import ipdb; ipdb.set_trace()
print(crossTable)

# Visualize the counts of target predictor by
plotTable = crossTable[crossTable.index != 'All']
plt.bar(plotTable.index, plotTable['Count'])
plt.xticks([[0], [1]])
plt.xlabel('CarOwnership')
plt.ylabel('Count')
plt.grid(True, axis='y')
# plt.show()

lease_count = crossTable.loc["Lease"]["Count"]
own_count = crossTable.loc["Own"]["Count"]
none_count = crossTable.loc["None"]["Count"]
total_count = crossTable.loc["All"]["Count"]

entropy_rn = -(((lease_count/total_count) * numpy.log2(lease_count/total_count)) + ((own_count/total_count) *
                                                                                    numpy.log2(own_count/total_count))
               + ((none_count/total_count) * numpy.log2(none_count/total_count)))

def entropy(crossTable):
    lease_count = crossTable.loc["Lease"]["Count"]
    own_count = crossTable.loc["Own"]["Count"]
    total_count = crossTable.loc["All"]["Count"]

    try:
        none_count = crossTable.loc["None"]["Count"]
    except KeyError:
        entropy_node = -(
                ((lease_count / total_count) * numpy.log2(lease_count / total_count)) + ((own_count / total_count) *
                                                                                         numpy.log2(
                                                                                             own_count / total_count))
                    )
    else:
        entropy_node = -(
                ((lease_count / total_count) * numpy.log2(lease_count / total_count)) + ((own_count / total_count) *
                                                                                         numpy.log2(
                                                                                             own_count / total_count))
                + ((none_count / total_count) * numpy.log2(none_count / total_count)))
    return entropy_node

def binary_split_generator(predictor):
    pred_vals = predictor.unique()
    binary_splits = []
    for index in range(len(pred_vals)):
        binary_splits.append([pred_vals[index]]) # filling list of splitted items in binary_split list
        for num_combo in range(1, len(pred_vals)-index -1):
            for item in list(itertools.combinations(range(index+1,len(pred_vals)), num_combo)):
                items = [pred_vals[index]]
                items = items + pred_vals[list(item)].tolist()
                binary_splits.append(items)

    for left_branch_index in range(len(binary_splits)):
        for right_branch_index in range(left_branch_index+1, len(binary_splits)): # looping against left splits
            # treating them as right branch
            #import ipdb;ipdb.set_trace()
            try:
                if sorted(binary_splits[left_branch_index] + binary_splits[right_branch_index]) == \
                        sorted(pred_vals.tolist()): # when the left branch and right branch makes the complete set,
                    # its redundant split
                    binary_splits.pop(right_branch_index)
            except IndexError:
                break
    return binary_splits

def entropy_table_generator(data,predictor):
    pred_vals = predictor.unique()
    dataTable = data
    b_splits = binary_split_generator(predictor)
    entropy_table = pandas.DataFrame(columns=["index","left_child_content","split_entropy"])
    # Now we move to calculate entropy of each split
    for i in range(len(b_splits)):
        left_child_crossTable = pandas.crosstab(index=dataTable[predictor.isin(b_splits[i])]['CarOwnership'],
                                                columns=["Count"], margins=True, dropna=True).drop(columns=['All'])
        #import ipdb; ipdb.set_trace()
        left_child_entropy = entropy(left_child_crossTable)

        right_child_crossTable = pandas.crosstab(index=dataTable[~predictor.isin(b_splits[i])]['CarOwnership'],
                                                columns=["Count"], margins=True, dropna=True).drop(columns=['All'])

        right_child_entropy = entropy(right_child_crossTable)
        left_child_count = left_child_crossTable.loc["All"]["Count"]
        right_child_count = total_count - left_child_count
        entropy_split = (left_child_count/total_count)*left_child_entropy + \
                        (right_child_count/total_count)*right_child_entropy
        # we will fill the index, entropy and split contents for each split in a dataframe
        index = i
        left_child_content = b_splits[i]
        #right_child_content =
        records = [index,left_child_content, entropy_split]
        entropy_table.loc[index,:]=records
    # entropy_table.set_index("index",inplace=True)
    return entropy_table

print("Entropy of Root Node is :", (entropy_rn))
print("There are %(n_node)d binary splits that we can generate from credit card predictor. "
      %{ "n_node": (pow(2,(len(CustomerSurvey_data['CreditCard'].unique()))-1)-1)} )

print(entropy_table_generator(CustomerSurvey_data,CustomerSurvey_data["CreditCard"]))
# printing the optimal branch/split
entropy_table_creditcard = entropy_table_generator(CustomerSurvey_data,CustomerSurvey_data["CreditCard"])
entropy_table_creditcard['split_entropy'] = entropy_table_creditcard['split_entropy'].astype('float64')
print(f"The optimal split is:\n {entropy_table_creditcard.iloc[entropy_table_creditcard['split_entropy'].idxmin(), :]}")

print("There are %(n_node)d binary splits that we can generate from credit card predictor. "
      %{ "n_node": (pow(2,(len(CustomerSurvey_data['JobCategory'].unique()))-1)-1)} )

#import ipdb;ipdb.set_trace()
print(entropy_table_generator(CustomerSurvey_data,CustomerSurvey_data["JobCategory"]))
# printing the optimal branch/split
entropy_table_jobcategory = entropy_table_generator(CustomerSurvey_data,CustomerSurvey_data["JobCategory"])
entropy_table_jobcategory['split_entropy'] = entropy_table_jobcategory['split_entropy'].astype('float64')
print(f"The optimal split is:\n {entropy_table_jobcategory.iloc[entropy_table_jobcategory['split_entropy'].idxmin(), :]}")