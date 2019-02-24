import itertools
import math
import pandas
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

#U_itemset = set({'A', 'B', 'C', 'D', 'E', 'F', 'G'})

print("The total number of possible itemsets are: %(k)d " %{"k":math.pow(2,7) -1})
def k_itemset_generator(k):
    return(list(itertools.combinations('ABCDEFG', k)))


print("All possible 1-itemsets are: %(liste)s" %{'liste': k_itemset_generator(1)})
print("All possible 2-itemsets are: %(liste)s" %{'liste': k_itemset_generator(2)})
print("All possible 3-itemsets are: %(liste)s" %{'liste': k_itemset_generator(3)})
print("All possible 4-itemsets are: %(liste)s" %{'liste': k_itemset_generator(4)})
print("All possible 5-itemsets are: %(liste)s" %{'liste': k_itemset_generator(5)})
print("All possible 6-itemsets are: %(liste)s" %{'liste': k_itemset_generator(6)})
print("All possible 7-itemsets are: %(liste)s" %{'liste': k_itemset_generator(7)})

################################## Analysing Association Rules in data using Apriori Algorithm of mlextend library######

Grocery_data = pandas.read_csv("Groceries.csv")
#print(Grocery_data)

# Number of customers
n_customer = Grocery_data.groupby(["Customer"])
print("There are %(n_cust)d number of customers in given market basket data." %{'n_cust':len(n_customer)})

# Number of unique items
n_item = Grocery_data.groupby(["Item"])
print("There are %(items)d number of unique items in given manket basket data."%{"items":len(n_item)})

# customer with unique item count dataset
n_customer_item = Grocery_data.groupby(["Customer"])["Item"].count()
print(n_customer_item)
print("The summary of required data containing unique number of items bought by customers is as follows:" )
print(n_customer_item.describe())

plt.figure()
ax = plt.gca
hist_data = plt.hist(n_customer_item, bins=n_customer_item.max()) #Taking maximum number of unique items bought by
# customer as number of bins to keep bin-width as one.
plt.title("Number of unique items versus Frequency")
#vertical line at 25th, median/50th and 75th percentile
plt.vlines(x=n_customer_item.quantile(q=0.25), ymin=0, ymax=2500, label="Q1", linestyles="solid")
plt.vlines(x=n_customer_item.quantile(q=0.5), ymin=0, ymax=2500, label="Median", linestyles="dashed")
plt.vlines(x=n_customer_item.quantile(q=0.75), ymin=0, ymax=2500, label="Q3", linestyles="dotted")
plt.legend()
plt.xlabel("Unique item counts")
plt.ylabel("Frequency")
plt.show()

# The threshhold will be 75/9385 = 0.008
ListItem = Grocery_data.groupby(['Customer'])['Item'].apply(list).values.tolist()
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)
# Median is taken as maximum itemset
frequent_itemsets = apriori(ItemIndicator, min_support = 75/(len(n_customer)), max_len = 3, use_colnames = True)
#import ipdb;ipdb.set_trace()
#print(frequent_itemsets)
print("The k-item sets which appeared in the market basket of at least seventy-five(75) customers are: \n",
      frequent_itemsets['itemsets'])
print("\nThe number of itemsets found are:",len(frequent_itemsets['itemsets']))
# Itemsets found are 522 and maximum K = 3 as observed by itemsets column of frequent_itemsets dataframe


# association rule for the frequent itemsets
assoc_rules = association_rules(frequent_itemsets, metric='confidence',min_threshold = 0.01) #default metric is confidence
print("There are %(n_assoc_rules)d association rules found for minimum confidence threshold of 1 percent."
      %{'n_assoc_rules':assoc_rules.shape[0]})
print(assoc_rules[['antecedents','consequents']])
# The graph of Support metrics on the vertical axis against the Confidence metrics on the horizontal axis for the rules
# found in with confidence threshold of 1%, using the Lift metrics to indicate the size of the marker.
plt.figure()
plt.scatter(assoc_rules['confidence'],assoc_rules['support'], s=assoc_rules['lift'], c=assoc_rules['lift'])  # s to indicate size of marker
plt.title("Graph of Support against Confidence metrics")
plt.xlabel("Confidence")
plt.ylabel("Support")
cbar = plt.colorbar()
cbar.set_label("Lift")
plt.show()


# association rule with 60% threshhold
assoc_rules = association_rules(frequent_itemsets, metric='confidence',min_threshold = 0.6) #default metric is confidence
print("There are %(n_assoc_rules)d association rules found for minimum confidence threshold of 60 percent."
      %{'n_assoc_rules':assoc_rules.shape[0]})
print(assoc_rules[['antecedents','consequents', 'antecedent support', 'consequent support', 'confidence','support','lift']])
# The similarity is presence of item "Whole-milk"


