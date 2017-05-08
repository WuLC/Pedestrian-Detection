# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2017-04-29 16:56:17
# @Last modified by:   WuLC
# @Last Modified time: 2017-05-07 14:47:27
# @Email: liangchaowu5@gmail.com

import io
import os

train_pos_path = './PR_dataset/train/pos/'
train_neg_path = './PR_dataset/train/neg/'
test_pos_path = './PR_dataset/test/pos/'
test_neg_path = './PR_dataset/test/neg/'

all_train_data = './PR_dataset/all_train_data'
train_data_1 = './PR_dataset/train_data_1'
train_data_2 = './PR_dataset/train_data_2'
test_data = './PR_dataset/test_data'


def check_files(dir_path):
    """check if the length of features of all files are the same
       the result is yes, the length is 2330, and the number of samples is listed below
       train_pos: 3605
       train_neg: 10055
       test_pos: 2043
       test_neg: 4832
    """
    files = os.listdir(dir_path)
    count = {}
    for file in files:
        f = dir_path+file
        print f
        with io.open(f, mode = 'r') as rf:
            lines = len(rf.readlines())
            count.setdefault(lines, 0)
            count[lines] += 1
    print count


def format_and_split_data():
    # construct test data
    if not os.path.exists(test_data):
        with io.open(test_data, mode = 'w', encoding = 'utf8') as wf:
            for test_path in (test_pos_path, test_neg_path):
                files = os.listdir(test_path)
                for file in files:
                    file = test_path+file
                    with io.open(file, mode = 'r') as rf:
                        content = ' '.join(map(lambda x: x.strip(), rf.readlines()))
                        # add label
                        if test_path == test_pos_path:
                            content = '1 '+content
                        elif test_path == test_neg_path:
                            content = '0 '+content
                        wf.write((content+'\n').decode('utf8'))
    print 'test data constructed, path: %s'%(test_data)

    # construct train data and split it into two parts
    if not (os.path.exists(train_data_1) and os.path.exists(train_data_2)):
        with io.open(all_train_data, mode = 'w', encoding = 'utf8') as wf:
            with io.open(train_data_1, mode = 'w', encoding = 'utf8') as wf1:
                with io.open(train_data_2, mode = 'w', encoding = 'utf8') as wf2:
                    pos_count, neg_count = 0, 0
                    for train_path in (train_pos_path, train_neg_path):
                        files = os.listdir(train_path)
                        for file in files:
                            file = train_path+file
                            with io.open(file, mode = 'r') as rf:
                                content = ' '.join(map(lambda x: x.strip(), rf.readlines()))
                                # add label
                                if train_path == train_pos_path:
                                    pos_count += 1
                                    content = '1 '+content
                                    wf.write((content+'\n').decode('utf8'))
                                    # split data
                                    if pos_count <= 1800:
                                        wf1.write((content+'\n').decode('utf8'))
                                    else:
                                        wf2.write((content+'\n').decode('utf8'))
                                elif train_path == train_neg_path:
                                    neg_count += 1
                                    content = '0 '+content
                                    wf.write((content+'\n').decode('utf8'))
                                    if neg_count <= 5025:
                                        wf1.write((content+'\n').decode('utf8'))
                                    else:
                                        wf2.write((content+'\n').decode('utf8'))
    print 'train data constructed, path: %s, %s, %s'%(all_train_data, train_data_1, train_data_2)


if __name__ == '__main__':
    for file in (train_pos_path, train_neg_path, test_pos_path, test_neg_path):
        check_files(file)
    format_and_split_data()