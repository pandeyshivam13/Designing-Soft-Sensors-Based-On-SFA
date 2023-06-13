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

    
# plt.plot(t,X['Flow 1(in cc/s)'])
# plt.plot(t,X['Flow 2(in cc/s)'])
# plt.plot(t,X['Flow 3(in cc/s)'])
# plt.plot(t,X['Flow 4(in cc/s)'])
# plt.plot(t,X['H1(cm)'])
# plt.plot(t,X['H2(cm)'])
# plt.plot(t,X['H3(cm)'])
plt.plot(t,Y)

data = np.vstack([X['Flow 1(in cc/s)'],X['Flow 2(in cc/s)'],X['Flow 3(in cc/s)'],X['Flow 4(in cc/s)'],X['H1(cm)'],X['H2(cm)'],X['H3(cm)']]).T

# Apply SFA to the data
sfa = SFA(n_components=4)
sfa.fit(data)
slow_features = sfa.transform(data)

# Print the slow features
# print(slow_features)
fig, ax = plt.subplots(3, 1, sharex=True)
fig.subplots_adjust(hspace=0.5)
ax[2].plot(slow_features)
X = slow_features
# print(type(X))
df = pd.DataFrame(X, columns=['A', 'B','C','D'])
X = df
# print(X)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 23)
X_train_0 = np.c_[np.ones((X_train.shape[0],1)),X_train]
X_test_0 = np.c_[np.ones((X_test.shape[0],1)),X_test]

theta = np.matmul(np.linalg.inv( np.matmul(X_train_0.T,X_train_0) ), np.matmul(X_train_0.T,Y_train)) 
# The parameters for linear regression model
parameter = ['theta_'+str(i) for i in range(X_train_0.shape[1])]
columns = ['intersect:x_0=1'] + list(X.columns.values)
parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':theta})
lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)
sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)
parameter_df = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))
y_pred_norm =  np.matmul(X_test_0,theta)
J_mse = np.sum((y_pred_norm - Y_test)**2)/ X_test_0.shape[0]

# R_square 
sse = np.sum((y_pred_norm - Y_test)**2)
sst = np.sum((Y_test - Y_test.mean())**2)
R_square = 1 - (sse/sst)
print('The Mean Square Error(MSE) or J(theta) is: ',J_mse)
print('R square obtain for normal equation method is :',R_square)

y_pred_sk = lin_reg.predict(X_test)

#Evaluvation: MSE
from sklearn.metrics import mean_squared_error
J_mse_sk = mean_squared_error(y_pred_sk, Y_test)

# R_square
R_square_sk = lin_reg.score(X_test,Y_test)
print('The Mean Square Error(MSE) or J(theta) is: ',J_mse_sk)
print('R square obtain for scikit learn library is :',R_square_sk)

f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.scatterplot(Y_test,y_pred_sk,ax=ax,color='r')
ax.set_title('Check for Linearity:\n Actual Vs Predicted value')
#NORMALISING TO 0 TO 1