# date 26 
import pandas as pd
import pickle as pk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from numpy import cov
from scipy.stats import pearsonr
from sksfa import SFA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV



read_file = pd.read_excel('./Data_Quadruple_Tank_System.xlsx')
read_file.to_csv('TrainData1.csv',index = None, header = True)
TrainingData = pd.read_csv('./TrainData1.csv')
TrainingData = TrainingData.drop(columns = ['S.No.','Flow 1','Flow 2','Flow 3','Flow 4','H1','H2','H3','H4'],axis = 1)
# print(TrainingData)
TrainingData['Flow 1(in cc/s)'] = TrainingData['Flow 1(in cc/s)'].fillna(TrainingData['Flow 1(in cc/s)'].mean())
TrainingData['Flow 2(in cc/s)'] = TrainingData['Flow 2(in cc/s)'].fillna(TrainingData['Flow 2(in cc/s)'].mean())
TrainingData['Flow 3(in cc/s)'] = TrainingData['Flow 3(in cc/s)'].fillna(TrainingData['Flow 3(in cc/s)'].mean())
TrainingData['Flow 4(in cc/s)'] = TrainingData['Flow 4(in cc/s)'].fillna(TrainingData['Flow 4(in cc/s)'].mean())
TrainingData['H1(cm)'] = TrainingData['H1(cm)'].fillna(TrainingData['H1(cm)'].mean())
TrainingData['H2(cm)'] = TrainingData['H2(cm)'].fillna(TrainingData['H2(cm)'].mean())
TrainingData['H3(cm)'] = TrainingData['H3(cm)'].fillna(TrainingData['H3(cm)'].mean())
TrainingData['H4(cm)'] = TrainingData['H4(cm)'].fillna(TrainingData['H4(cm)'].mean())

X = TrainingData.drop(columns = 'H4(cm)',axis = 1)
# print(X)
Y = TrainingData['H4(cm)']
# len(X)
# print(Y)
t = np.linspace(0, 2045, 2046)

#NORMALISATION 
for i in X.index:
    X['Flow 1(in cc/s)'][i] = (X['Flow 1(in cc/s)'][i]-X['Flow 1(in cc/s)'].min())/(X['Flow 1(in cc/s)'].max()-X['Flow 1(in cc/s)'].min())
    X['Flow 2(in cc/s)'][i] = (X['Flow 2(in cc/s)'][i]-X['Flow 2(in cc/s)'].min())/(X['Flow 2(in cc/s)'].max()-X['Flow 2(in cc/s)'].min())
    X['Flow 3(in cc/s)'][i] = (X['Flow 3(in cc/s)'][i]-X['Flow 3(in cc/s)'].min())/(X['Flow 3(in cc/s)'].max()-X['Flow 3(in cc/s)'].min())
    X['Flow 4(in cc/s)'][i] = (X['Flow 4(in cc/s)'][i]-X['Flow 4(in cc/s)'].min())/(X['Flow 4(in cc/s)'].max()-X['Flow 4(in cc/s)'].min())
    X['H1(cm)'][i] = (X['H1(cm)'][i]-X['H1(cm)'].min())/(X['H1(cm)'].max()-X['H1(cm)'].min())
    X['H2(cm)'][i] = (X['H2(cm)'][i]-X['H2(cm)'].min())/(X['H2(cm)'].max()-X['H2(cm)'].min())
    X['H3(cm)'][i] = (X['H3(cm)'][i]-X['H3(cm)'].min())/(X['H3(cm)'].max()-X['H3(cm)'].min())
    Y[i] = (Y[i]-Y.min())/(Y.max()-Y.min())

fig1, ax1 = plt.subplots()
ax1.plot(t,X['Flow 1(in cc/s)'],color='b', label='Flow 1')
ax1.plot(t,X['Flow 2(in cc/s)'],color='g', label='Flow2')
ax1.plot(t,X['Flow 3(in cc/s)'],color='r',label='Flow 3')
ax1.plot(t,X['Flow 4(in cc/s)'],color='y',label='Flow 4')
ax1.plot(t,X['H1(cm)'],color='k',label='H1')
ax1.plot(t,X['H2(cm)'],color='c',label='H2')
ax1.plot(t,X['H3(cm)'],color='m',label='H3')
ax1.legend(bbox_to_anchor =(0.25, 1.0))   
# plt.plot(t,X['Flow 1(in cc/s)'])
# plt.plot(t,X['Flow 2(in cc/s)'])
# plt.plot(t,X['Flow 3(in cc/s)'])
# plt.plot(t,X['Flow 4(in cc/s)'])
# plt.plot(t,X['H1(cm)'])
# plt.plot(t,X['H2(cm)'])
# plt.plot(t,X['H3(cm)'])
data = np.vstack([X['Flow 1(in cc/s)'],X['Flow 2(in cc/s)'],X['Flow 3(in cc/s)'],X['Flow 4(in cc/s)'],X['H1(cm)'],X['H2(cm)'],X['H3(cm)']]).T

# Apply SFA to the data
sfa = SFA(n_components=3)
sfa.fit(data)
slow_features = sfa.transform(data)

# Print the slow features
# print(slow_features)
fig, ax = plt.subplots(3, 1, sharex=True)
fig.subplots_adjust(hspace=0.5)
ax[2].plot(slow_features)
X = slow_features
# print(type(X))
df = pd.DataFrame(X, columns=['A', 'B','C'])
X = df
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_Y = StandardScaler()
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# X_train = sc_X.fit_transform(X_train)
# Y_train = np.squeeze(sc_Y.fit_transform(Y_train.reshape(-1, 1)))

# from sklearn.svm import SVR
# regressor = SVR(kernel = 'rbf')
# regressor.fit(X_train, Y_train) 
# y_pred = regressor.predict([X_test])
# y_pred = sc_Y.inverse_transform(y_pred)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

sse = np.sum((y_pred - y_test)**2)
sst = np.sum((y_test - y_test.mean())**2)
R_square = 1 - (sse/sst)
mse = sse/len(y_test)
print('The Mean Square Error(MSE) or J(theta) is: ',mse)
print('R square obtain for normal equation method is :',R_square)