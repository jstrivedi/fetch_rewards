import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression() :
    def __init__( self, learning_rate = 0.00002, iterations=1000000) :
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def fit(self, X, Y):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(Y, np.ndarray):
            Y = Y.to_numpy()
            
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        for i in range(self.iterations):
            self.update_weights()
        return self
            
    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = -(2*(self.X.T).dot(self.Y-Y_pred))/self.m
        db = -2*np.sum(self.Y-Y_pred)/self.m 
        self.W = self.W-self.learning_rate*dW
        self.b = self.b-self.learning_rate*db
        return self

    def predict(self,X):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        return X.dot(self.W) + self.b

def meansqerror(y,y_pred): # y_pred and y needs to be a numpy array
    return np.sqrt(np.mean((y_pred-y)**2))

data = pd.read_csv('data_daily.csv')
data['# Date'] = pd.to_datetime(data['# Date'])
data['Receipt_Count'] = data['Receipt_Count'].astype('float')
data = data.set_index('# Date')
data['x'] = range(1, data.shape[0]+1)

weekcol = []
for i in data.index.weekday:
    row = [0]*7
    row[i] = 1
    weekcol.append(row)


data2 = pd.concat([data, pd.DataFrame(np.array(weekcol), index = data.index, dtype = 'float')], axis = 1)
data3 = pd.DataFrame(data2.x.values+365, index = data2.index+pd.DateOffset(days=365))
data3 = data3.rename(columns={0:'x'})


reg   = LinearRegression()

#we bring in the weight and bias value from the training script
reg.W = np.array([7121.52868447])
reg.b = 7523237.690740455

print('\nLinear Regression Model Configurations:')
print('Weight/s : %s\nbias     : %s'%(reg.W, reg.b))

plt.figure(figsize=(20,2))
plt.title('Raw Data of 2021 and Prediction for 2022')
plt.plot(data2[['Receipt_Count']], label= "True")
plt.plot(data3.index, reg.predict(data3[['x']]), label= "predicted")
plt.legend()
plt.show()

index2mnth = {1:'January  ', 2:'February ', 3:'March    ', 4:'April    ', 5:'May      ', 6:'June     ',
              7:'July     ', 8:'August   ', 9:'September', 10:'October  ', 11:'November ', 12:'December '}

data3_pred = pd.DataFrame(reg.predict(data3[['x']]), index = data3.index)
mnth2count = {}
print('\nMonthwise Predictions for 2022')
for i in range(1,13):
    print('%s %.1f'%(index2mnth[i], data3_pred[data3_pred.index.month == i][0].sum()))
    mnth2count[index2mnth[i]] = data3_pred[data3_pred.index.month == i][0].sum()
    
plt.figure(figsize=(20,5))
plt.title('Barplot for Monthwise Predictions for 2022')
plt.bar(mnth2count.keys(), mnth2count.values())
plt.show()
print('\n')