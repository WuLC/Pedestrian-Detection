# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2017-04-29 16:54:08
# @Last modified by:   WuLC
# @Last Modified time: 2017-05-09 00:12:35
# @Email: liangchaowu5@gmail.com

import os
import io
import math

import cPickle as pickle
import numpy as np

from LogisticRegression import logistic_regression, sigmoid


all_train_data = './PR_dataset/all_train_data'
train_data_1 = './PR_dataset/train_data_1'
train_data_2 = './PR_dataset/train_data_2'
test_data = './PR_dataset/test_data'


def adaBoost(train_data, iterations , gd_iteration, train_method):
    data = train_data.split('/')[-1]
    adaboost_model_path = './models/adaboost_%s_lr_%s_%s_on_%s'%(iterations, train_method, gd_iteration, data)
    if not os.path.exists(adaboost_model_path):
        # count the number of samples, initial weights of them
        weights = []
        with io.open(train_data, mode = 'r', encoding = 'utf8') as rf:
            line = rf.readline()
            while line:
                weights.append(1.0)
                line = rf.readline()
        weights = np.array(weights)
        weights = 1.0*weights/weights.sum()

        models = []
        for m in xrange(iterations):
            w = logistic_regression(train_data_path = train_data, iterations = gd_iteration, train_method = train_method, sample_weights = weights)
            # caculate error rate and alpha
            with io.open(train_data, mode = 'r', encoding = 'utf8') as rf:
                line = rf.readline()
                idx, error_rate = 0, 0
                predict_correct = [True for i in xrange(len(weights))]
                while line:
                    fields = map(lambda x: float(x), line.strip().split())
                    y = fields[0]
                    fields[0] = 1.0 # bias item
                    x = np.array(fields)
                    predict = 1 if sigmoid(np.dot(w, x)) >= 0.5 else 0
                    if predict != y:
                        error_rate += weights[idx]
                        predict_correct[idx] = False
                    idx += 1
                    line = rf.readline()
            # exit when erroe rate is > 0.5
            if error_rate > 0.5:
                break
            alpha = 0.5*math.log(1.0*(1-error_rate)/error_rate)
            # update weight
            for i in xrange(len(weights)):
                if predict_correct[i]:
                    weights[i] *= math.exp(-alpha)
                else:
                    weights[i] *= math.exp(alpha)
            weights = 1.0*weights/weights.sum()
            # store current model
            print '========finish %s models, current model error rate: %s, current alpha: %s========'%(m+1, error_rate, alpha)
            models.append((alpha, w))
        pickle.dump(models, open(adaboost_model_path, 'w'))
    return adaboost_model_path



def predict(test_data_file, model_path):
    if not os.path.exists(model_path):
        print 'model not ready'
        return

    models = pickle.load(open(model_path))
    TP, FP, TN, FN = 0, 0, 0, 0
    with io.open(test_data_file, mode = 'r', encoding = 'utf8') as rf:
        line = rf.readline()
        count = 1
        while line:
            fields = map(lambda x: float(x), line.strip().split())
            y = fields[0]
            fields[0] = 1.0 # bias item
            x = np.array(fields)
            target_sum = 0
            for alpha, w in models:
                target_sum += alpha * sigmoid(np.dot(w, x))
            predict = 1 if target_sum >= 0.5 else 0
            line = rf.readline()
            # evaluate predicting result
            if y == 1:
                if predict == y:
                    TP += 1
                else:
                    FN += 1
            if y == 0:
                if predict == y:
                    TN += 1
                else:
                    FP += 1
    print 'TP:%s, FN:%s, TN:%s, FP:%s'%(TP, FN, TN, FP)           
    precision, recall = 1.0*TP/(TP+FP), 1.0*TP/(TP+FN)
    print "precision: %s, recall: %s"%(precision, recall)
    return precision, recall



if __name__ == '__main__':
    # customize parameters
    iterations =  10
    gd_iteration = 800
    train_method = 'bgd'
    
    print '=============training on train data 1======================'
    model_path_1 = adaBoost(train_data_1, iterations = iterations, gd_iteration = gd_iteration, train_method = train_method, )
    print '=============training on train data 2======================'
    model_path_2 = adaBoost(train_data_2, iterations = iterations, gd_iteration = gd_iteration, train_method = train_method, )
    
    print '=============performance on train data 1======================'
    p1, r1 = predict(train_data_1, model_path_2)
    print '=============performance on train data 2======================'
    p2, r2 = predict(train_data_2, model_path_1)
    print '=============average performance on train data 1 and 2======================'
    p, r = (p1+p2)/2.0, (r1+r2)/2.0
    print "precision: %s, recall: %s"%(p, r)



"""
adaboost: iteration: 10
LR: bgd, iteration: 200, step size = 0.01 
=============performance on train data 1======================
TP:1510, FN:290, TN:4399, FP:626
precision: 0.706928838951, recall: 0.838888888889
=============performance on train data 2======================
TP:1737, FN:68, TN:3319, FP:1711
precision: 0.503770301624, recall: 0.962326869806
=============average performance on train data 1 and 2======================
precision: 0.605349570288, recall: 0.900607879347


adaboost: iteration: 15
LR: bgd, iteration: 200, step size = 0.01
=============performance on train data 1======================
TP:1359, FN:441, TN:4483, FP:542
precision: 0.714886901631, recall: 0.755
=============performance on train data 2======================
TP:1655, FN:150, TN:3945, FP:1085
precision: 0.60401459854, recall: 0.916897506925
=============average performance on train data 1 and 2======================
precision: 0.659450750085, recall: 0.835948753463


adaboost: iteration: 20
LR: bgd, iteration: 200, step size = 0.01
=============performance on train data 1======================
TP:1610, FN:190, TN:4182, FP:843
precision: 0.656339176519, recall: 0.894444444444
=============performance on train data 2======================
TP:1715, FN:90, TN:3176, FP:1854
precision: 0.480526758196, recall: 0.950138504155
=============average performance on train data 1 and 2======================
precision: 0.568432967357, recall: 0.9222914743




adaboost: iteration: 10
LR: bgd, iteration: 500, step size = 0.01
=============performance on train data 1======================
TP:1659, FN:141, TN:4106, FP:919
precision: 0.643522110163, recall: 0.921666666667
=============performance on train data 2======================
TP:1772, FN:33, TN:3077, FP:1953
precision: 0.475704697987, recall: 0.981717451524
=============average performance on train data 1 and 2======================
precision: 0.559613404075, recall: 0.951692059095


adaboost: iteration: 10
LR: bgd, iteration: 800, step size = 0.01
=============performance on train data 1======================
TP:1601, FN:199, TN:4398, FP:627
precision: 0.718581687612, recall: 0.889444444444
=============performance on train data 2======================
TP:1419, FN:386, TN:4642, FP:388
precision: 0.785279468733, recall: 0.786149584488
=============average performance on train data 1 and 2======================
precision: 0.751930578172, recall: 0.837797014466


adaboost: iteration: 10
LR: bgd, iteration: 1000, step size = 0.01, with regularization
=============performance on train data 1======================
TP:1667, FN:133, TN:4073, FP:952
precision: 0.636502481863, recall: 0.926111111111
=============performance on train data 2======================
TP:1784, FN:21, TN:2849, FP:2181
precision: 0.449936948298, recall: 0.98836565097
=============average performance on train data 1 and 2======================
precision: 0.54321971508, recall: 0.95723838104


=========================================
        result on test data
=========================================
adaboost: iteration: 10
LR: bgd, iteration: 800, step size = 0.01, with regularization
TP:1694, FN:349, TN:4119, FP:713
precision: 0.703780639801, recall: 0.82917278512
=========================================
"""

