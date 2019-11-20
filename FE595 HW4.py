import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston, load_wine


## LINEAR REGRESSION

# Load the Boston dataset
boston = load_boston()
# Process the dataset into a data frame
fname = boston.feature_names
Boston = pd.DataFrame(boston.data, columns=fname)
Boston['MEDV'] = boston.target
print(Boston.head())
X = Boston.loc[:,'CRIM':'LSTAT']
y = Boston['MEDV']
# Do the linear regression
lr = LinearRegression
reg = lr().fit(X=X,y=y)
print(reg.coef_)
# Calculate the absolute of the coefficients
res = [abs(ele) for ele in reg.coef_]
# Find the element having the most influence
ele_ind = res.index(max(res))
EMI = fname[ele_ind]  # Element with the most influence
print('The element with the most influence is ' + str(EMI))

## K MEANS

# Load wine dataset
wine = load_wine()
# Process the data
w_name = wine.feature_names
Wine = pd.DataFrame(wine.data, columns=w_name)
print(Wine.head())
ssd = {}   # Sum of squared distances
for k in range(1,10):
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster = kmeans.fit(Wine)
    ssd[k] = cluster.inertia_
sns.pointplot(list(ssd.keys()),list(ssd.values()))
plt.title('The sum of squared distance'); plt.xlabel('Number of clusters'); plt.ylabel('SSD')
plt.show()
Assumed_number = len(list(wine.target_names))
print(Assumed_number)

# Results:
# As we could observe in the plot, 3 is the best number of the clusters. Also 3 is assumed cluster number.


