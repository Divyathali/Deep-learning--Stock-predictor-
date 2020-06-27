
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR 

import warnings
warnings.filterwarnings("ignore")


options = " Stock Linear Regression Prediction, Stock Logistic Regression Prediction, Support Vector Regression, Exit".split(",")

# Input Start Date
def start_date():
    date_entry = input('Enter a starting date in MM/DD/YYYY format: ')
    start = datetime.datetime.strptime(date_entry,'%m/%d/%Y')
    start = start.strftime('%Y-%m-%d')
    return start

# Input End Date
def end_date():
    date_entry = input('Enter a starting date in MM/DD/YYYY format: ')
    end = datetime.datetime.strptime(date_entry,'%m/%d/%Y')
    end = end.strftime('%Y-%m-%d')
    return end

# Input Symbols
def input_symbol():
    symbol = input("Enter symbol: ").upper()
    return symbol

def get_historical_data(symbol,start_date,end_date):
    symbol = symbol.upper()
    start = datetime.date(int(start_date[0:4]), int(start_date[5:7]), int(start_date[8:10]))
    end = datetime.date(int(end_date[0:4]), int(end_date[5:7]), int(end_date[8:10]))
    
    col_names = ['Date','Open','High','Low','Close','Volume']
    stocks = pd.read_csv('C:/Users/Admin/Desktop/stack market price/Stock-Price-Predictor-master/google.csv', header=0, names=col_names) 
    
    df = pd.DataFrame(stocks)
    return df




def stock_logistic_regression():
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = pd.read_csv('C:/Users/Admin/Desktop/stack market price/Deep-Learning-Machine-Learning-Stock-master/Stock_Apps/google.csv')
 
    df = df.drop(['Date'], axis=1)
    X = df.loc[:, df.columns != 'Adj Close']
    y = np.where (df['Close'].shift(-1) > df['Close'],1,-1)

    split = int(0.7*len(df))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    model = LogisticRegression()
    model = model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    print(metrics.confusion_matrix(y_test, predicted))
    print(metrics.classification_report(y_test, predicted))
    print(model.score(X_test,y_test))
    cross_val = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    print(cross_val)
    print(cross_val.mean())
    return

# Linear Regression
def stock_linear_regression():
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = pd.read_csv('C:/Users/Admin/Desktop/stack market price/Deep-Learning-Machine-Learning-Stock-master/Stock_Apps/google.csv')
 
    n = len(df.index)
    X = np.array(df['Open']).reshape(n,-1)
    Y = np.array(df['Close']).reshape(n,-1)
    lr = LinearRegression()
    lr.fit(X, Y)
    lr.predict(X)
    
    plt.figure(figsize=(12,8))
    plt.scatter(df['Close'], lr.predict(X))
    plt.plot(X, lr.predict(X), color = 'red')
    plt.xlabel('Prices')
    plt.ylabel('Predicted Prices')
    plt.grid()
    plt.title(sym + ' Prices vs Predicted Prices')
    plt.show()
    print('Summary:')       
    print('Estimate intercept coefficient:', lr.intercept_)
    print('Number of coefficients:', len(lr.coef_))
    print('Accuracy Score:', lr.score(X, Y))
    return

# Support Vector Regression
def stock_svr():
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = pd.read_csv('C:/Users/Admin/Desktop/stack market price/Deep-Learning-Machine-Learning-Stock-master/Stock_Apps/google.csv')
 
    dates = np.reshape(df.index,(len(df.index), 1)) # convert to 1xn dimension
    x = 31
    x = np.reshape(x,(len(x), 1))
    prices = df['Close']
    svr_lin  = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    # Fit regression model
    svr_lin .fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    plt.figure(figsize=(12,8))
    plt.scatter(dates, prices, c='k', label='Data')
    plt.plot(dates, svr_lin.predict(dates), c='g', label='Linear model')
    plt.plot(dates, svr_rbf.predict(dates), c='r', label='RBF model')    
    plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    print('Linear Model:', svr_rbf.predict(x)[0])
    print('RBF Model:', svr_lin.predict(x)[0])
    print('Polynomial Model:', svr_poly.predict(x)[0])
    return

    
def main():
    run_program = True
    while run_program:
     
        print("Choose Options:")
        for i in range(1, len(options)+1):
            print("{} - {}".format(i, options[i-1]))
        choice = int(input())
        
        if choice == 1:
            print("____________Linear Regression_____________")
            stock_linear_regression()
        elif choice == 2:
            print("____________Logistic Regression_____________")
            stock_logistic_regression()
        elif choice == 3:
            print("____________Support Vector Regression_____________")
            stock_logistic_regression()    
        elif choice == 4:
            run_program = False             


if __name__ == "__main__":
    main()
