import pandas
import numpy
from scipy.stats import iqr
import matplotlib.pyplot as plt
from numpy import percentile

#reading data from normal sample csv
normalsampledata = pandas.read_csv("Data/NormalSample.csv")

#print(normalsampledata.x)
#
print("IQR is : ",iqr(normalsampledata['x']))
IQR = iqr(normalsampledata['x'])

# According to Izenman method the recommended bin-width will be 2*IQR*N^-1/3

print("The bin-width for histogram of x according to the Izenman method is : ", 2*IQR*(1001**(-1/3)))


print("Minimum of x is : ", numpy.min(normalsampledata['x']))
print("Maximum of X is : ", numpy.max(normalsampledata['x']))


# a = 26 and b = 36


def density_cord_n_draw_hist(h):
    plt.figure()
    ax = plt.gca()
    bin_edges_list = []
    a=26.0
    m = round(a+h/2, 2)
    mid_point_list = []
    while(a<=36.0):
        bin_edges_list.append(a)
        mid_point_list.append(m)
        a=round(a+h,2)
        m=round(m+h,2)

    print("Bin edges:", bin_edges_list)
    mid_point_list = mid_point_list[:-1]
    hist_data = plt.hist(normalsampledata['x'], bins=bin_edges_list, density=True, color='green')  #density is set 1
    # so that its divided by N*h(number of observations times bin-width)
    ax.set_title(f"Histogram with bin-width {h}")
    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    desitites_list = hist_data[0] #collecting the desities or the Y values of histogram in a list
    for i in range(len(mid_point_list)):
        print(mid_point_list[i],desitites_list[i])  #prints the coordinates of mid-points and respective
        # density estimate


density_cord_n_draw_hist(0.1)
density_cord_n_draw_hist(0.5)
density_cord_n_draw_hist(1)
density_cord_n_draw_hist(2)
plt.show()


quartiles = percentile(normalsampledata['x'],[25,50,75])
# print("The five number summary is: \n min: ",numpy.min(normalsampledata.x)," Q1:", quartiles[0]," median:",
#       quartiles[1]," Q3:", quartiles[2], " and max:", numpy.max(normalsampledata.x))

data_descriptive = normalsampledata['x'].describe()
print("The summary of data is as follows:\n", data_descriptive)
print("The value of 1.5times IQR whikers are ",quartiles[0]-1.5*IQR, "and ", quartiles[2]+1.5*IQR)


x_group_0_data = normalsampledata.groupby(['group']).get_group(0)['x']
x_group_1_data = normalsampledata.groupby(['group']).get_group(1)['x']

group_0_data_descriptive = x_group_0_data.describe()
group_1_data_descriptive = x_group_1_data.describe()
print("The summary of group 0 data : ",group_0_data_descriptive)
print("The summary of group 1 data : ",group_1_data_descriptive)

print(" The value of 1.5IQR whikers for group 0 category is \n",
      group_0_data_descriptive['25%']-1.5*iqr(x_group_0_data),
      "\n and \n",
      "32.20")  # as the 1.5times IQR+ Q3 is more than max of group 0 data so maximum will be right side whisker of 1.5IQR


print(" The value of (1.5times - Q1) whikers for group 1 category is \n",
      group_1_data_descriptive['25%']-1.5*iqr(x_group_1_data),
      "\n The value of (1.5times + Q3) whikers for group 1 category is \n",
      group_1_data_descriptive['75%']+1.5*iqr(x_group_1_data))


normalsampledata.boxplot(column='x',vert=False)
ax =plt.gca()
ax.grid(False)
plt.title('Boxplot of x')
plt.xlabel("x")
plt.vlines(x=quartiles[0]-1.5*IQR,ymin=0,ymax=50,colors="Purple",linestyles='solid' )
plt.vlines(x=quartiles[2]+1.5*IQR,ymin=0,ymax=50,colors="Purple",linestyles='solid' )
plt.show()


plt.figure()
ax = plt.gca()
ax.boxplot([normalsampledata['x'], x_group_0_data, x_group_1_data], vert=False)
ax.set_yticklabels(['x', 'x grp 0', 'x grp 1'])
plt.title('Boxplot of x for each category')
plt.suptitle("")
plt.xlabel("x")
plt.show()

x_outside_group_0_leftwhisk = normalsampledata['x'][normalsampledata['group']==0][normalsampledata['x'] <
                                                                group_0_data_descriptive['25%']-1.5*iqr(x_group_0_data)]
x_outside_group_0_rightwhisk = normalsampledata['x'][normalsampledata['group']==0][normalsampledata['x'] >
                                                                32.20]

x_outside_group_1_leftwhisk =normalsampledata['x'][normalsampledata['group']==1][normalsampledata['x'] <
                                                                group_1_data_descriptive['25%']-1.5*iqr(x_group_1_data)]
x_outside_group_1_rightwhisk = normalsampledata['x'][normalsampledata['group']==1][normalsampledata['x'] >
                                                                group_1_data_descriptive['75%']+1.5*iqr(x_group_1_data)]

x_outside_leftwhisk= normalsampledata['x'][normalsampledata['x']<data_descriptive['25%']-1.5*IQR]
x_outside_rightwhisk= normalsampledata['x'][normalsampledata['x']>data_descriptive['75%']+1.5*IQR]

print("The outliers of group 0 is: \n", pandas.concat([x_outside_group_0_leftwhisk,x_outside_group_0_rightwhisk]))
print("The outliers of group 1 is: \n", pandas.concat([x_outside_group_1_leftwhisk, x_outside_group_1_rightwhisk]))
print("The overall outliers of x is: \n", pandas.concat([x_outside_leftwhisk,x_outside_rightwhisk]))
