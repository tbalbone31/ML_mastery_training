## %%
# Import modules
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
import numpy


# %%
# Load data and set cols
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)

# %%
description = data.describe()
# %%
# Histogram of each attribute
data.hist()

# %%
data.plot(kind='box')

# %%
# Scatterplot Matrix
scatter_matrix(data)
plt.show()

#%%
# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
# create numpy array from dataframe
array = data.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])

#%%
# Noramlise using a range option (0-1)
from sklearn.preprocessing import MinMaxScaler
# create numpy array from dataframe
array = data.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
# Create scaler objects and fit/transform
minmax_scaler = MinMaxScaler()
minmax_scaledX = minmax_scaler.fit_transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(minmax_scaledX[0:5,:])

#%%
# Mapping to a Uniform distribution
from sklearn.preprocessing import QuantileTransformer

# create numpy array from dataframe
array = data.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
# Create transformation objects and fit/transform
quantile_transformer = QuantileTransformer(random_state=0)
quantile_transformedX = quantile_transformer.fit_transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(quantile_transformedX[0:5,:])



# %%

