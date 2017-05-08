# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2017-04-26 14:58:12
# @Last Modified by:   WuLC
# @Last Modified time: 2017-05-07 21:12:46

import os
import io
import math

import cPickle as pickle
import numpy as np


all_train_data = './PR_dataset/all_train_data'
train_data_1 = './PR_dataset/train_data_1'
train_data_2 = './PR_dataset/train_data_2'
test_data = './PR_dataset/test_data'
                            

def sigmoid(x):
    try:
        res = 1 / (1 + math.exp(-x))
    except OverflowError:
        res = 1 if x>0 else 0
    return res


def logistic_regression(train_data_path, step_size= 0.01, iterations = 100, train_method = 'sgd', lambda_ = 0.1, sample_weights = None):
    # construct path of model
    data = train_data_path.split('/')[-1]
    model_path = './models/lr_%s(%s, %s)_on_%s'%(train_method, iterations, step_size, data)
    if not os.path.exists('./models/'):
        os.mkdir('./models/')
    if sample_weights != None or not os.path.exists(model_path):
        w = np.zeros([1, 2331]) # length of features plus one for the bias
        if train_method == 'sgd': # train with sgd
            old_w = None
            for i in xrange(iterations):
                if i > 0:
                    diff = np.abs(w - old_w).sum()
                    print 'sgd, %s iterations, coefficients abs difference: %s' %(i+1, diff)
                old_w = np.array(w)
                with io.open(train_data_path, mode = 'r', encoding = 'utf8') as rf:
                    idx = 0
                    line = rf.readline()
                    while line:
                        fields = map(lambda x: float(x), line.strip().split())
                        y = fields[0]
                        fields[0] = 1.0 # bias item
                        x = np.array(fields)
                        if sample_weights != None:  # adaboost
                            w += step_size*sample_weights[idx]*(y - sigmoid(np.dot(w, x)))*x
                            idx += 1
                        else:
                            w += step_size*(y - sigmoid(np.dot(w, x)))*x
                        line = rf.readline()
        elif train_method == 'bgd':  # train with bgd
            for i in xrange(iterations):
                if i > 0:
                    diff = np.abs(w - old_w).sum()
                    print 'bgd, %s iterations, coefficients abs difference: %s' %(i+1, diff)
                old_w = np.array(w)
                offset = np.zeros([1, 2331])
                with io.open(train_data_path, mode = 'r', encoding = 'utf8') as rf:
                    line = rf.readline()
                    idx = 0
                    while line:
                        fields = map(lambda x: float(x), line.strip().split())
                        y = fields[0]
                        fields[0] = 1.0 # bias item
                        x = np.array(fields)
                        if sample_weights != None:
                            offset += sample_weights[idx]*(y - sigmoid(np.dot(w, x)))*x
                            idx += 1
                        else:
                            offset += (y - sigmoid(np.dot(w, x)))*x
                        line = rf.readline()
                # regularization
                offset += 2 * lambda_ * w
                w += step_size * offset
        else:
            print 'train method %s is not available '%(train_method)
            return 
        
        # don't persist on the model for adaboost method, just return the weights
        if sample_weights != None:
            return w 
        pickle.dump(w, open(model_path, 'w'))
    return model_path


def predict(test_data_file, model_path):
    if not os.path.exists(model_path):
        print 'model not ready'
        return
    w = pickle.load(open(model_path))[0]
    TP, FP, TN, FN = 0, 0, 0, 0
    with io.open(test_data_file, mode = 'r', encoding = 'utf8') as rf:
        line = rf.readline()
        count = 1
        while line:
            fields = map(lambda x: float(x), line.strip().split())
            y = fields[0]
            fields[0] = 1.0 # bias item
            x = np.array(fields)
            predict = 1 if sigmoid(np.dot(w, x)) >= 0.5 else 0
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
    
    # customized parameters
    step_size = 0.01
    iterations =  1500
    train_method = 'bgd' # sgd or bgd
    lambda_ = 0.1  # parameter for regularization
    
    print '=============training on train data 1======================'
    model_path_1 = logistic_regression(train_data_1, step_size = step_size, iterations = iterations, train_method = train_method, lambda_ = lambda_)
    print '=============training on train data 2======================'
    model_path_2 = logistic_regression(train_data_2, step_size = step_size, iterations = iterations, train_method = train_method, lambda_ = lambda_)
    
    print '=============performance on train data 1======================'
    p1, r1 = predict(train_data_1, model_path_2)
    print '=============performance on train data 2======================'
    p2, r2 = predict(train_data_2, model_path_1)
    print '=============average performance on train data 1 and 2======================'
    p, r = (p1+p2)/2.0, (r1+r2)/2.0
    print "precision: %s, recall: %s"%(p, r)
    """
    print '=============training on all data ======================'
    model_path = logistic_regression(all_train_data, step_size = step_size, iterations = iterations, train_method = train_method, lambda_ = lambda_)
    p, r = predict(test_data, model_path)
    print "precision: %s, recall: %s"%(p, r)
    """




"""
==============================================================
                    result on train data
==============================================================

sgd, iteration = 200, step_size = 0.01
=============performance on train data 1======================
TP:133, FN:1667, TN:5000, FP:25
precision: 0.841772151899, recall: 0.0738888888889
=============performance on train data 2======================
TP:144, FN:1661, TN:5022, FP:8
precision: 0.947368421053, recall: 0.0797783933518
=============average performance on train data 1 and 2======================
precision: 0.894570286476, recall: 0.0768336411203

sgd, iteration = 200, step size = 0.05
=========performance on train data 1======================
TP:185, FN:1615, TN:4993, FP:32
precision: 0.852534562212, recall: 0.102777777778
=============performance on train data 2======================
TP:126, FN:1679, TN:5011, FP:19
precision: 0.868965517241, recall: 0.0698060941828
=============average performance on train data 1 and 2======================
precision: 0.860750039727, recall: 0.0862919359803

sgd, iteration = 200, step_size = 0.1
=============performance on train data 1======================
TP:148, FN:1652, TN:4997, FP:28
precision: 0.840909090909, recall: 0.0822222222222
=============performance on train data 2======================
TP:157, FN:1648, TN:5007, FP:23
precision: 0.872222222222, recall: 0.0869806094183
=============average performance on train data 1 and 2======================
precision: 0.856565656566, recall: 0.0846014158203


sgd, iteration = 500, step_size = 0.01
=============performance on train data 1======================
TP:175, FN:1625, TN:4997, FP:28
precision: 0.862068965517, recall: 0.0972222222222
=============performance on train data 2======================
TP:311, FN:1494, TN:4985, FP:45
precision: 0.873595505618, recall: 0.172299168975
=============average performance on train data 1 and 2======================
precision: 0.867832235568, recall: 0.134760695599


sgd, iteration = 800, step_size = 0.01
=============performance on train data 1======================
TP:380, FN:1420, TN:4962, FP:63
precision: 0.857787810384, recall: 0.211111111111
=============performance on train data 2======================
TP:321, FN:1484, TN:4976, FP:54
precision: 0.856, recall: 0.17783933518
=============average performance on train data 1 and 2======================
precision: 0.856893905192, recall: 0.194475223146


sgd, iteration = 1000, step_size = 0.01
=============performance on train data 1======================
TP:397, FN:1403, TN:4958, FP:67
precision: 0.855603448276, recall: 0.220555555556
=============performance on train data 2======================
TP:351, FN:1454, TN:4968, FP:62
precision: 0.849878934625, recall: 0.194459833795
=============average performance on train data 1 and 2======================
precision: 0.85274119145, recall: 0.207507694675


sgd, iteration = 1500, step_size = 0.01
=============performance on train data 1======================
TP:436, FN:1364, TN:4944, FP:81
precision: 0.84332688588, recall: 0.242222222222
=============performance on train data 2======================
TP:471, FN:1334, TN:4946, FP:84
precision: 0.848648648649, recall: 0.260941828255
=============average performance on train data 1 and 2======================
precision: 0.845987767264, recall: 0.251582025239

sgd, iteration = 2000, step_size = 0.01
=============performance on train data 1======================
TP:490, FN:1310, TN:4933, FP:92
precision: 0.841924398625, recall: 0.272222222222
=============performance on train data 2======================
TP:509, FN:1296, TN:4941, FP:89
precision: 0.851170568562, recall: 0.281994459834
=============average performance on train data 1 and 2======================
precision: 0.846547483594, recall: 0.277108341028
#####################################################################################


#####################################################################################
bgd, iteration = 200, step_size = 0.01
=============performance on train data 1======================
TP:1699, FN:101, TN:3576, FP:1449
precision: 0.539707750953, recall: 0.943888888889
=============performance on train data 2======================
TP:794, FN:1011, TN:4893, FP:137
precision: 0.852846401719, recall: 0.439889196676
=============average performance on train data 1 and 2======================
precision: 0.696277076336, recall: 0.691889042782

bgd, iteration = 200, step_size = 0.01, L2 regularization:
=============performance on train data 1======================
TP:1727, FN:73, TN:3366, FP:1659
precision: 0.510041346722, recall: 0.959444444444
=============performance on train data 2======================
TP:1350, FN:455, TN:4636, FP:394
precision: 0.774082568807, recall: 0.747922437673
=============average performance on train data 1 and 2======================
precision: 0.642061957765, recall: 0.853683441059


bgd, iteration = 200, step_size = 0.05
=======performance on train data 1======================
TP:1699, FN:101, TN:3576, FP:1449
precision: 0.539707750953, recall: 0.943888888889
=============performance on train data 2======================
TP:794, FN:1011, TN:4893, FP:137
precision: 0.852846401719, recall: 0.439889196676
=============average performance on train data 1 and 2======================
precision: 0.696277076336, recall: 0.691889042782


bgd, iteration = 200, step size = 0.1
=============performance on train data 1======================
TP:1699, FN:101, TN:3576, FP:1449
precision: 0.539707750953, recall: 0.943888888889
=============performance on train data 2======================
TP:794, FN:1011, TN:4893, FP:137
precision: 0.852846401719, recall: 0.439889196676
=============average performance on train data 1 and 2======================
precision: 0.696277076336, recall: 0.691889042782


bgd, iteration = 500, step_size = 0.01
=============performance on train data 1======================
TP:1603, FN:197, TN:4350, FP:675
precision: 0.703687445127, recall: 0.890555555556
=============performance on train data 2======================
TP:237, FN:1568, TN:4983, FP:47
precision: 0.834507042254, recall: 0.131301939058
=============average performance on train data 1 and 2======================
precision: 0.76909724369, recall: 0.510928747307

bgd, iteration = 500, step_size = 0.01, with L2 regularization
=============performance on train data 1======================
TP:1437, FN:363, TN:4624, FP:401
precision: 0.781828073993, recall: 0.798333333333
=============performance on train data 2======================
TP:1339, FN:466, TN:4711, FP:319
precision: 0.807599517491, recall: 0.741828254848
=============average performance on train data 1 and 2======================
precision: 0.794713795742, recall: 0.77008079409


bgd, iteration = 800, step_size = 0.01
=============performance on train data 1======================
TP:1595, FN:205, TN:4410, FP:615
precision: 0.721719457014, recall: 0.886111111111
=============performance on train data 2======================
TP:1423, FN:382, TN:4681, FP:349
precision: 0.803047404063, recall: 0.78836565097
=============average performance on train data 1 and 2======================
precision: 0.762383430538, recall: 0.83723838104

bgd, iteration = 800, step_size = 0.01, with L2 regularization
=============performance on train data 1======================
TP:1446, FN:354, TN:4624, FP:401
precision: 0.782891174878, recall: 0.803333333333
=============performance on train data 2======================
TP:1435, FN:370, TN:4640, FP:390
precision: 0.786301369863, recall: 0.795013850416
=============average performance on train data 1 and 2======================
precision: 0.784596272371, recall: 0.799173591874


bgd, iteration = 1000, step_size = 0.01
=============performance on train data 1======================
TP:1626, FN:174, TN:4345, FP:680
precision: 0.705117085863, recall: 0.903333333333
=============performance on train data 2======================
TP:1287, FN:518, TN:4787, FP:243
precision: 0.841176470588, recall: 0.713019390582
=============average performance on train data 1 and 2======================
precision: 0.773146778226, recall: 0.808176361958

bgd, iteration = 1000, step_size = 0.01, with L2 regularization
=============performance on train data 1======================
TP:1479, FN:321, TN:4566, FP:459
precision: 0.763157894737, recall: 0.821666666667
=============performance on train data 2======================
TP:1456, FN:349, TN:4618, FP:412
precision: 0.779443254818, recall: 0.806648199446
=============average performance on train data 1 and 2======================
precision: 0.771300574777, recall: 0.814157433056


bgd, iteration = 1500, step_size = 0.01
=============performance on train data 1======================
TP:1525, FN:275, TN:4574, FP:451
precision: 0.771761133603, recall: 0.847222222222
=============performance on train data 2======================
TP:660, FN:1145, TN:4950, FP:80
precision: 0.891891891892, recall: 0.365650969529
=============average performance on train data 1 and 2======================
precision: 0.831826512748, recall: 0.606436595876

bgd, iteration = 1500, step_size = 0.01, with L2 regularization
=============performance on train data 1======================
TP:1476, FN:324, TN:4574, FP:451
precision: 0.765957446809, recall: 0.82
=============performance on train data 2======================
TP:1467, FN:338, TN:4593, FP:437
precision: 0.770483193277, recall: 0.812742382271
=============average performance on train data 1 and 2======================
precision: 0.768220320043, recall: 0.816371191136


bgd, iteration = 2000, step_size = 0.01
=============performance on train data 1======================
TP:1730, FN:70, TN:3788, FP:1237
precision: 0.583080552747, recall: 0.961111111111
=============performance on train data 2======================
TP:1356, FN:449, TN:4721, FP:309
precision: 0.814414414414, recall: 0.751246537396 
=============average performance on train data 1 and 2======================
precision: 0.698747483581, recall: 0.856178824254


bgd, iteration = 2000, step_size = 0.01, with L2 regularization
=============performance on train data 1======================
TP:1482, FN:318, TN:4571, FP:454
precision: 0.765495867769, recall: 0.823333333333
=============performance on train data 2======================
TP:1467, FN:338, TN:4590, FP:440
precision: 0.76927110645, recall: 0.812742382271
=============average performance on train data 1 and 2======================
precision: 0.767383487109, recall: 0.818037857802



==========================================================================
                    result on test data
============================================================================
bgd, iteration: 800, step_size = 0.01, with regularization 
TP:1534, FN:509, TN:4425, FP:407
precision: 0.790314270994, recall: 0.750856583456
"""