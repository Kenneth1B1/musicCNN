from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
import pandas as pd

NB = GaussianNB()


def preprocess():
    names = []
    dataset = pd.read_csv("music.csv")
    dataset = dataset.drop(['filename'], axis = 1)
    
    #create train and testing data
    array = dataset.values
    X = array[:,16:58]
    Y = array[:,58]
    validation_size = 0.55
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = validation_size, random_state = 1, shuffle = True)
    return X_train, X_test, Y_train, Y_test
    
    print("processed")

def train(X_train, X_test, Y_train, Y_test):

    NB.fit(X_train, Y_train)
    predictions = NB.predict(X_test)
    print(accuracy_score(Y_test,predictions))
    #kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    #cv_results = cross_val_score(NB, X_train, Y_train, cv=kfold, scoring='accuracy')
    
    #print('%s: %f (%f)' % ("NB", cv_results.mean(), cv_results.std()))


def test():

    #prediction = NB.predict([X[1]])
    return "Blues"

    
if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = preprocess()
    train(X_train, X_test, Y_train, Y_test)
    print(test())
