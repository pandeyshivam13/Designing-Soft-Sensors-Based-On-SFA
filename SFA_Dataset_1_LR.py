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

read_file = pd.read_excel('./Sample_Data_btp1.xlsx')
read_file.to_csv('TrainData.csv',index = None, header = True)
TrainingData = pd.read_csv('./TrainData.csv')
# print(TrainingData)
sum_disso = 0.0 
sum_temp = 0.0
sum_stemp = 0.0
sum_t = 0.0
n1 = 0
n2 = 0
n3 = 0
n4 = 0
for i in TrainingData.index:
    if((float)(TrainingData['Dissolved oxygen (ppmw)'][i])!=0.0):
        sum_disso += (float)(TrainingData['Dissolved oxygen (ppmw)'][i])
        n1 += 1
    if(65.0 <= (float)(TrainingData['Fluid temperature (oC)'][i]) <= 95.0):
        sum_temp += (float)(TrainingData['Fluid temperature (oC)'][i])
        n2 += 1
    if(150.0 <= (float)(TrainingData['Surface temperature (oC)'][i]) <= 450.0):
        sum_stemp += (float)(TrainingData['Surface temperature (oC)'][i])
        n3 += 1
    if(15.0 <= (float)(TrainingData['Time (hr)'][i]) <= 48.0):
        sum_t += (float)(TrainingData['Time (hr)'][i])
        n4 += 1
#equivalent diameter is irrelevant 
TrainingData = TrainingData.drop(columns = ['Equivalent diameter (m)','Fluid velocity (m/s)'],axis = 1)
avg_dis = sum_disso/n1 
avg_temp = sum_temp/n2 
avg_stemp = sum_stemp/n3
avg_t = sum_t/n4 
for i in TrainingData.index:
    if((float)(TrainingData['Dissolved oxygen (ppmw)'][i])==0.0):
        (TrainingData['Dissolved oxygen (ppmw)'][i]) = avg_dis
    if(not(65.0 <= (float)(TrainingData['Fluid temperature (oC)'][i]) <= 95.0)):
        (TrainingData['Fluid temperature (oC)'][i]) = avg_temp
    if(not(150.0 <= (float)(TrainingData['Surface temperature (oC)'][i]) <= 450.0)):
        (TrainingData['Surface temperature (oC)'][i]) = avg_stemp
    if(not(15.0 <= (float)(TrainingData['Time (hr)'][i]) <= 48.0)):
        (TrainingData['Time (hr)'][i]) = avg_t
        
X = TrainingData.drop(columns = 'Fouling factor (m2 K/kW)',axis = 1)
Y = TrainingData['Fouling factor (m2 K/kW)']
t = np.linspace(0, 1388, 1389)

#min-max
mean_do = X['Dissolved oxygen (ppmw)'].min()
mean_de = X['Density (Kg/m3)'].min()
mean_t = X['Time (hr)'].min()
mean_st = X['Surface temperature (oC)'].min()
mean_ft = X['Fluid temperature (oC)'].min()
mean_y = Y.min()

std_do = X['Dissolved oxygen (ppmw)'].max()
std_de = X['Density (Kg/m3)'].max()
std_t = X['Time (hr)'].max()
std_st = X['Surface temperature (oC)'].max()
std_ft = X['Fluid temperature (oC)'].max()
std_y = Y.max()

for i in X.index:
    X['Dissolved oxygen (ppmw)'][i] = (X['Dissolved oxygen (ppmw)'][i]-mean_do)/(std_do-mean_do)
    X['Density (Kg/m3)'][i] = (X['Density (Kg/m3)'][i]-mean_de)/(std_de-mean_de)
    X['Time (hr)'][i] = (X['Time (hr)'][i]-mean_t)/(std_t-mean_t)
    X['Surface temperature (oC)'][i] = (X['Surface temperature (oC)'][i]-mean_st)/(std_st-mean_st)
    X['Fluid temperature (oC)'][i] = (X['Fluid temperature (oC)'][i]-mean_ft)/(std_ft-mean_ft)
    Y[i] = (Y[i]-mean_y)/(std_y-mean_y)

fig1, ax1 = plt.subplots()
ax1.plot(t,X['Dissolved oxygen (ppmw)'],color='b', label='Dissolved oxygen')
ax1.plot(t,X['Density (Kg/m3)'],color='g', label='Density')
ax1.plot(t,X['Time (hr)'],color='r',label='Time (hr)')
ax1.plot(t,X['Surface temperature (oC)'],color='y',label='Surface temperature (oC)')
ax1.plot(t,X['Fluid temperature (oC)'],color='k',label='Fluid temperature')
ax1.legend(bbox_to_anchor =(0.25, 1.0))

data = np.vstack([X['Dissolved oxygen (ppmw)'],X['Density (Kg/m3)'],X['Time (hr)'],X['Surface temperature (oC)'],X['Fluid temperature (oC)']]).T

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