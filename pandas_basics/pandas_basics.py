import pandas as pd
import numpy as np


# important documents
# https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
# https://stackoverflow.com/questions/31593201/how-are-iloc-ix-and-loc-different

# iloc - selecting or indexing based on position - row number or col number
# loc - selecting or indexing by name - row name / col name
# ix indexing can be done both by position and name , in pandas version 0.20.0 and above, ix is deprecated

#Create a DataFrame
data = {
    'Name':['Alisa','Bobby','Cathrine','Alisa','Bobby','Cathrine',
            'Alisa','Bobby','Cathrine','Alisa','Bobby','Cathrine'],
    'Exam':['Semester 1','Semester 1','Semester 1','Semester 1','Semester 1','Semester 1',
            'Semester 2','Semester 2','Semester 2','Semester 2','Semester 2','Semester 2'],

    'Subject':['Mathematics','Mathematics','Mathematics','Science','Science','Science',
               'Mathematics','Mathematics','Mathematics','Science','Science','Science'],
   'Score':[62,47,55,74,31,77,85,63,42,67,89,81]}

example_df = pd.DataFrame(data)
print(example_df)
first_two_rows = example_df.iloc[:2]
print(first_two_rows)
first_two_rows = example_df.iloc[:2,]
print(first_two_rows)
third_to_fifth = example_df.iloc[2:5]
print(third_to_fifth)
third_to_fifth = example_df.iloc[2:5,]
print(third_to_fifth)

all_from_third = example_df.iloc[2:]
print(all_from_third)
all_from_third = example_df.iloc[2:,]
print(all_from_third)

first_two_cols = example_df.iloc[:,:2]
print(first_two_cols)

first_n_fourth_cols = example_df.iloc[:,[0,3]]
print(first_n_fourth_cols)

second_row_n_third_col = example_df.iloc[1,2]
print(" The value in second row and third column: ", second_row_n_third_col)

# select using row name using loc

row_1 = example_df.loc[1]
print(" The row value in row 1 ", row_1)
row_n_col_label = example_df.loc[[1,2,3,4,5],["Name","Score"]]

print("Selecting row and col ",row_n_col_label)