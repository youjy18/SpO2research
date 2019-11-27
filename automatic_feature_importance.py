# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import pandas as pd
# load data
dataframe = pd.read_excel('input_all3.xlsx',header=1 )
dataset = dataframe.values
X = dataset[:,1:6]
y = dataset[:,6]
# fit model on training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()