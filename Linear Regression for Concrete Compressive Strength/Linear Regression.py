
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# from sklearn.linear_model import Ridge


def Get_corrcoef(data):
    return np.corrcoef(data.values.T)

def heatmap_corrcoef(cm,cols):
    hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',yticklabels = cols,xticklabels= cols)
    plt.tight_layout()
    plt.savefig('heatmap_new.png',dpi=300)

def standar_data(X):
    sc_x = StandardScaler()
    X_std = sc_x.fit_transform(X)
    return X_std

def split_data(X,y):
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
    return x_train,x_test,y_train,y_test



cols = ['Cement','BFS','FA','Water','Spl','CA','FAgg','Age','CCS']
data = pd.read_excel('Concrete_Data.xls')
data.columns = cols



#算出9個變數間的相關係數
cm = Get_corrcoef(data)
heatmap_corrcoef(cm,data.columns.values.tolist())
print('得變數間相關係數的矩陣:','\n',cm,'\n','-'*100)

#線性迴歸模型

x_data =  data.drop(['CCS','FA','FAgg','CA'], axis=1).values
y_data = data['CCS'].values

x_data_std = standar_data(x_data)


x_train,x_test,y_train,y_test = split_data(x_data_std,y_data)


print('切割完畢，訓練集數量:',x_train.shape,' 測試集:',x_test.shape,'\n','-'*100)



LR = LinearRegression()
quadratic = PolynomialFeatures(degree=4)
X_train = quadratic.fit_transform(x_train)
X_test = quadratic.fit_transform(x_test)
LR.fit(X_train,y_train)
# print('訓練完畢，得到迴歸係數:\n',LR.coef_,'\n','-'*100)
train_pred = LR.predict(X_train)
test_pred = LR.predict(X_test)
print('MSE train : %2f | MSE test: %2f' % (mean_squared_error(y_train,train_pred),mean_squared_error(y_test,test_pred)))
print('R^2 train : %2f | R^2 test: %2f' % (r2_score(y_train,train_pred),r2_score(y_test,test_pred)))



#for testing
'''
for i in range(1,10):
    LR = LinearRegression()
    quadratic = PolynomialFeatures(degree=i)
    X_train = quadratic.fit_transform(x_train)
    X_test = quadratic.fit_transform(x_test)
    LR.fit(X_train,y_train)

    # print('訓練完畢，得到迴歸係數:\n',LR.coef_,'\n','-'*100)
    train_pred = LR.predict(X_train)
    test_pred = LR.predict(X_test)
    print('-------------degree:',i,'-------------')
    print('MSE train : %2f | MSE test: %2f' % (mean_squared_error(y_train,train_pred),mean_squared_error(y_test,test_pred)))
    print('R^2 train : %2f | R^2 test: %2f' % (r2_score(y_train,train_pred),r2_score(y_test,test_pred)))

'''

