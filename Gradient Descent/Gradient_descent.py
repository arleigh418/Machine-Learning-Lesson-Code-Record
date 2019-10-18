import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#E(θ0, θ1) =1/2∑(hθ(x^(i)) − y^(i))^2
def cost_function(x,y,parser1,parser2,lr): 
    h0 = parser1 + parser2*x
    h_update1 = h0-y
    h_update2 = (h0-y)*x
    return h_update1 , h_update2

#count new theda0(θ0) and theda1(θ1)
def theda(parser1,parser2,lr,h0_update,h1_update):
    parser1_update = parser1 - lr*add(h0_update) 
    parser2_update = parser2 - lr*add(h1_update)
    return parser1_update,parser2_update

# Regression f(x) = theda0 + theda1*x
def count(parser1,parser2,x,y):
    result = []
    for i in range(len(x)):
        result.append(parser1+parser2*x[i])
    return result

def count_cost(parser1,parser2,x,y):
    error = (y - (parser1+parser2*x))**2
    return error

def add(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum

#read file
data = pd.read_csv('regression.csv')
x = data['X'].tolist()
y = data['y'].tolist()
lr = 0.01

#for cost
parser_cost = []

#running
for epoch in range(0,200):
    cost = 0
    h_update1_list = []
    h_update2_list = []
    for i in range(len(x)):
        if epoch == 0:
            parser1 = 0 #first theda0 = 0
            parser2 = 0 #first theda1 = 0
            h_update1,h_update2 = cost_function(x[i],y[i],parser1,parser2,lr) #count each with x data
            h_update1_list.append(h_update1)
            h_update2_list.append(h_update2)
            cost+=count_cost(parser1,parser2,x[i],y[i])
        else:
            parser1 = parser1_update
            parser2 = parser2_update
            h_update1,h_update2 = cost_function(x[i],y[i],parser1,parser2,lr)
            h_update1_list.append(h_update1)
            h_update2_list.append(h_update2)
            cost+=count_cost(parser1,parser2,x[i],y[i])

    parser_cost.append(cost/len(x))
    parser1_update,parser2_update= theda(parser1,parser2,lr,h_update1_list,h_update2_list) #new update theda0 theda1
    print('第',epoch,'次訓練更新','parser1:',parser1_update,'parser2:',parser2_update)


#result visualization
ag=list(range(200))
result = count(parser1_update,parser2_update,x,y)

fig = plt.figure()
plt.scatter(x,y)
plt.title('Function and Target Y')
plt.xlabel('x label')
plt.ylabel('Y label')
plt.plot(x,result)
plt.savefig('Function_target.png')
plt.close()


fig2 = plt.figure()
plt.title('Cost - Each epoch')
plt.xlabel('epoch')
plt.ylabel('Cost')
plt.plot(ag,parser_cost,'g-',label = 'Cost')
plt.legend(loc='best')
plt.savefig('cost.png')

