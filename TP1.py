# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:20:19 2020

@author: Tiago Botelho 52009 Bernardo Martins 53292
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def plot_errors(validation_error, train_error, plot_name, fig_name):
    
    plt.title(plot_name)
    
    plt.plot(validation_error, 'r', label = 'Validation Error')
    plt.plot(train_error, 'b', label = 'Training Error')
    
    plt.legend(loc='upper right', shadow = True)
    plt.savefig(fig_name, dpi = 300)
    plt.close()
    
    return

###################################
####### Logistic Regression #######
def calc_fold_lr(X,Y, train_ix,valid_ix,C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix,:],Y[train_ix])
    
    prob = reg.predict_proba(X[:,:])[:,1]
    squares = (prob-Y)**2
    
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])

def logistic_regression(n_folds, x_train, y_train, x_test, y_test):
    
    kf = StratifiedKFold(n_splits = n_folds)

    best_c = 100000000
    best_error = 100000
    
    validation_error = []
    train_error = []
    
    c_range = [10**i for i in range(-2, 13)]
    
    for c in c_range:
        t_error = v_error = 0
        for train_ix, valid_ix in kf.split(y_train, y_train):
            t, v = calc_fold_lr(x_train, y_train, train_ix, valid_ix, c)
            t_error += t
            v_error += v
        
        v_error = v_error/n_folds
        t_error = t_error/n_folds
        
        train_error.append(t_error)
        validation_error.append(v_error)
        
        if(v_error < best_error):
            best_error = v_error
            best_c = c
           
    strLg = "The best C value is: {}"
    print(strLg.format(best_c))
    reg = LogisticRegression(C=best_c, tol=1e-10)
    reg.fit(x_train,y_train) 
    test_error = 1 - reg.score(x_test, y_test)  #erro medio calculado, fazemos a subtraçao para calcular o erro
    rg = reg.predict(x_test)
    
    plot_errors(validation_error, train_error, 'Logistic Regression', 'LR.png')
    
    return test_error, rg


###############################
####### Our Naïve Bayes #######
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score


def split_data(feature, classes):
    f0 = feature[classes==0]
    f1 = feature[classes==1]
    return f0, f1

def kernel(F,bw):
    kde = KernelDensity(kernel='gaussian', bandwidth=bw)
    kde.fit(F)
    return kde

def nbfit(features, classes, bw):
    kernels = []
    
    priori0 = np.log(features[classes==0,:].shape[0]/classes.shape[0]) #Probabilidade de pertencer à classe 0
    priori1 = np.log(features[classes==1,:].shape[0]/classes.shape[0]) #Probabilidade de pertencer à classe 1
    for i in range(0,4):
        f0, f1 = split_data(features[:,i], classes) #Para cada feature
        k0 = kernel(f0.reshape(-1,1), bw)
        k1 = kernel(f1.reshape(-1,1), bw)
        kernels.append((k0,k1))
    return priori0, priori1, kernels

def nbpred(p0, p1, kernels, X, Y):
    pred = np.zeros(X.shape[0])
    for i in range(0,4):
        #Somatório dos logaritmos
        p0 += kernels[i][0].score_samples(X[:,[i]])
        p1 += kernels[i][1].score_samples(X[:,[i]])
    for j in range(X.shape[0]):
        if p0[j] < p1[j]: #argmax
            pred[j] = 1
    return 1 - accuracy_score(Y, pred), pred

#kernel density estimator porque temos valores  continuos ver onde eles se concentram
def calc_foldNB(X,Y, train_ix,valid_ix,bw):
    p0, p1, kernels = nbfit(X[train_ix,:4],Y[train_ix], bw)
    r, pred = nbpred(p0, p1, kernels, X[train_ix,:4], Y[train_ix])
    
    p0, p1, kernels_Validation = nbfit(X[valid_ix,:4],Y[valid_ix], bw)
    v, pred = nbpred(p0, p1, kernels, X[valid_ix,:4], Y[valid_ix])
    return r, v

def naive_bayes(folds, x_train, y_train, x_test, y_test):
    min = 1
    bw = 0.02
    train_err = []
    valid_err = []
    kf = StratifiedKFold(n_splits = folds)

    while bw <= 0.6:
        tr_err = va_err = 0
        for train,test in kf.split(y_train, y_train):
            r,v = calc_foldNB(x_train,y_train,train,test,bw)
            tr_err += r
            va_err += v
            
        if va_err/folds < min:
            min = va_err/folds
            bestBw = bw
        train_err.append(tr_err/folds)
        valid_err.append(va_err/folds)
        bw+=0.02
        
    strLg = "The best bandwidth is: {}"
    print(strLg.format(bestBw))
    p0, p1, kernels = nbfit(x_train, y_train, bestBw)
    x, nb = nbpred(p0, p1, kernels, x_test, y_test)
    test_error = 1 - accuracy_score(y_test, nb)
    plot_errors(valid_err, train_err, 'Naive Bayes', 'NB.png')
    return test_error, nb


####################################
####### Gaussian Naïve Bayes #######
from sklearn.naive_bayes import GaussianNB
    
def gaussian_naive_bayes(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    
    test_error_nb_gaussian = 1 - accuracy_score(y_test, y_pred)
    return test_error_nb_gaussian, y_pred


##########################
####### Normal Test#######
import math

def AproxNormalTest(N, p0):
    X = N*p0
    v = math.sqrt(N*(X/N)*(1-(X/N)))
    
    return X, 1.96*v


#############################
####### McNemar's Test#######
def auxCompare(pred1, pred2, Ys):
    res = 0
    for i in range(0, len(Ys)):
        if pred1[i]!=Ys[i] and pred2[i]==Ys[i]:
            res+=1
        
    return res

def McNemar(e1, e2):
    mod = abs(e1 - e2)
    res = ((mod-1)**2) / (e1+e2)
    return res
  

####### Process the data correctly, including randomizing the order of the data points and standardizing the values #######
def stddata(File1, File2):
    #load and standardize data
    dataTrain = np.loadtxt(File1,delimiter='\t')
    dataTest = np.loadtxt(File2,delimiter='\t')
    np.random.shuffle(dataTrain)
    
    YsTrain = dataTrain[:,-1]
    XsTrain = dataTrain[:,:-1]
    
    means = np.mean(XsTrain,axis=0)
    stdevs = np.std(XsTrain,axis=0)
    XsTrain = (XsTrain-means)/stdevs
    
    YsTest = dataTest[:,-1]
    XsTest = dataTest[:,:-1]
    
    XsTest = (XsTest-means)/stdevs
    
    return XsTrain, YsTrain, XsTest, YsTest

def main():
    Xs_train, Ys_train, Xs_test, Ys_test = stddata("TP1_train.tsv", "TP1_test.tsv")
    
    test_error_lr, rg = logistic_regression(5, Xs_train, Ys_train, Xs_test, Ys_test)
    print('----------Logistic Regression----------')
    print(test_error_lr)
    print()
    
    test_error_nb, nb = naive_bayes (5, Xs_train, Ys_train, Xs_test, Ys_test)
    print('--------------Naive Bayes--------------')
    print(test_error_nb)
    print()
    
    test_error_nb_gaussian, gnb = gaussian_naive_bayes(Xs_train, Ys_train, Xs_test, Ys_test)
    print('---------Gaussian Naive Bayes----------')
    print(test_error_nb_gaussian)
    print()
    
    #Normal Test
    LG_T, v_LG = AproxNormalTest(len(Xs_test), test_error_lr)
    NB_T, v_NB = AproxNormalTest(len(Xs_test), test_error_nb)
    GNB_T, v_GNB = AproxNormalTest(len(Xs_test), test_error_nb_gaussian)
    
    print("--------------Normal Test--------------")
    strLg = "The Logistic Regresion Interval is: {} +/- {}"
    print(strLg.format(LG_T, v_LG))
    strLg = "The Naive Bayes Interval is: {} +/- {}"
    print(strLg.format(NB_T, v_NB))
    strLg = "The Gaussian Naive Bayes Interval is: {} +/- {}"
    print(strLg.format(GNB_T, v_GNB))
    
    print()
    
    #McNemar's
    LGvsNB = auxCompare(rg, nb, Ys_test)   
    NBvsLG = auxCompare(nb, rg, Ys_test)
    
    LGvsGNB = auxCompare(rg, gnb, Ys_test)   
    GNBvsLG = auxCompare(gnb, rg, Ys_test)
    
    NBvsGNB = auxCompare(nb, gnb, Ys_test)   
    GNBvsNB = auxCompare(gnb, nb, Ys_test)
    
    res_LGvsNB = McNemar(LGvsNB, NBvsLG)
    res_LGvsGNB = McNemar(LGvsGNB, GNBvsLG)
    res_NBvsGNB = McNemar(NBvsGNB, GNBvsNB)
    print("---------------McNemar's---------------")
    print("Logistic Regression vs Naive Bayes: ", res_LGvsNB)
    print("Logistic Regression vs Gaussian Naive Bayes: ", res_LGvsGNB)
    print("Naive Bayes vs Gaussian Naive Bayes: ", res_NBvsGNB)
    
main()