#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/28 9:49
# @Author  : zhuojian chen
# @File    : dsep.py
# @Description : Code for paper “Distributed sequential estimation procedures"

from joblib import dump
import numpy as np
from scipy import stats
import scipy
import math
from random import sample
import scipy.linalg
import time
import pandas as pd
import os
from datetime import datetime
import multiprocessing as mp
import csv
import copy


def rmse(a, b):
    '''
    :param a: vector
    :param b: vector
    :return: rmse value between vector a and b
    '''
    sum = 0
    error = a - b
    for i in range(len(error)):
        sum += error[i] ** 2
    return math.sqrt(sum / len(error))


# 获取最大特征值
def lam_max(x):
    '''
    :param x: matrix
    :return: max eigenvalue of matrox x
    '''
    e, v = np.linalg.eig(x)
    return np.max(e)


def gen_data(params):
    '''
    :param params:
    :return: training and testing datasets
    '''
    N = params['N']
    p = params['p']
    rho = params['rho']
    center = 0.2  # center
    sigma = np.zeros((p - 1, p - 1))
    for i in range(p - 1):
        for j in range(p - 1):
            sigma[i, j] = rho ** (abs(i - j))
    X = stats.multivariate_normal.rvs(mean=(np.zeros(p - 1) + center), cov=sigma, size=int(N))
    X = np.hstack((np.ones(np.shape(X)[0]).reshape(np.shape(X)[0], 1), X))
    beta0 = params['beta0']
    theta = np.dot(X, beta0)
    Y = []
    for i in range(len(theta)):
        Y.append(theta[i] + np.random.normal(loc=0, scale=1.0, size=1))
    data_mat = np.hstack([X, np.array(Y)])
    test_id = sample(range(params['N']), int(params['test_ratio'] * params['N']))
    train_id = list(set(range(params['N'])) - set(test_id))
    train = data_mat[train_id, :]
    test = data_mat[test_id, :]
    return ({'train_id': train_id, 'train': train, 'test': test})


class init_data(object):
    def __init__(self, params):
        self.p = params['p']
        tmp = gen_data(params)
        self.train_id = tmp['train_id']
        self.data = tmp['train']  # training data
        self.test = tmp['test']
        index = range(np.shape(self.data)[0])
        self.init_ids = sample(index, params['init_N'])
        self.labeled_ids = self.init_ids
        self.unlabeled_ids = list(set(index) - set(self.labeled_ids))
        self.train = self.data[self.labeled_ids, :]
        self.new = 0


def LSE(X, Y):
    '''
    :param X: covariates
    :param Y: response
    :return:
    '''
    X = np.array(X)
    n = np.shape(X)[0]
    p = np.shape(X)[1]
    beta_est = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    # beta_est = stats.linregress(X,Y)
    res = Y - X.dot(beta_est)
    sigma2 = res.dot(res.T) / (n - p)
    mu = lam_max(n * np.linalg.inv(X.T.dot(X)))
    return {'beta': beta_est, 'sigma2': sigma2, 'mu': mu}


def D_optimal(params, data_env):
    '''
    :param params:
    :param data_env:
    :return: select data id according to D-optimality
    '''
    data = data_env.data
    unlabeled_ids = data_env.unlabeled_ids
    p = params['p']
    A = data_env.train[:, :p].T.dot(data_env.train[:, :p])
    W_n = np.linalg.inv(A)
    det = []
    for ind in unlabeled_ids:
        x = data[ind, :p]
        det.append(x.dot(W_n).dot(x.T))
    ind = unlabeled_ids[det.index(max(det))]
    return (ind)


def is_stopped(data_env, params):
    '''
    :param data_env:
    :param params:
    :return: whether the stopping condition is satisfied
    '''
    n = len(data_env.train)
    d = params['d']
    sigma2 = params['sigma2']
    p = params['p']
    mu = params['mu']
    K = params['K']
    return (sigma2 + 1 / n <= d ** 2 * n * K / (scipy.stats.chi2.ppf(0.95, p) * mu))


def update_data(data_env, ind):
    newdata = data_env.data[ind, :]
    data_env.labeled_ids.append(ind)
    data_env.unlabeled_ids.remove(ind)
    data_env.train = np.vstack([data_env.train, newdata])
    return data_env


def distributed_seq_ana(data_env, params, adaptive='random', method='lse'):
    '''
    :param data_env:
    :param params:
    :param adaptive: 'random' or 'D_optimal'
    :param method: 'lse'
    :return: related simulation results
    '''
    K = params['K']
    p = params['p']
    n_all = len(data_env.data)
    res_list = []
    beta0 = params['beta0']
    for i in range(K):
        start_time = time.time()
        data_env_new = copy.deepcopy(data_env)
        paramsnew = params.copy()
        index = range(int(i * n_all / K), int((i + 1) * n_all / K))
        labels_index = sample(index, paramsnew['init_N'])
        unlabels_index = list(set(index) - set(labels_index))
        data_env_new.labeled_ids = labels_index
        data_env_new.unlabeled_ids = unlabels_index
        data_env_new.train = data_env_new.data[labels_index]
        i = len(data_env_new.labeled_ids)
        while True:
            X = data_env_new.train[:, range(data_env_new.p)]
            Y = data_env_new.train[:, data_env_new.p]
            if (method == 'lse'):
                result = LSE(X, Y)
            elif (method == 'lsp'):
                result = LSP(X, Y, paramsnew)
            elif (method == 'ase'):
                result = ASE(X, Y, paramsnew)
            paramsnew['beta_est'] = result['beta']
            paramsnew['sigma2'] = result['sigma2']
            paramsnew['mu'] = result['mu']
            if (not is_stopped(data_env_new, paramsnew)):
                if (adaptive == 'random'):
                    ind = sample(data_env_new.unlabeled_ids, 1)[0]
                elif (adaptive == 'D_optimal'):
                    ind = D_optimal(paramsnew, data_env_new)
                elif (adaptive == 'A_optimal'):
                    ind = A_optimal(paramsnew, data_env_new)
                update_data(data_env_new, ind)
                i = i + 1
            else:
                paramsnew['N'] = i + 1
                res = get_info(paramsnew, data_env_new)
                res['method'] = method
                res['adaptive'] = adaptive
                end_time = time.time()
                res['time'] = end_time - start_time
                break
        res_list.append(res)
    N = 0
    for i in range(K):
        N += res_list[i]['N']
    beta_est = 0
    label_ids = []
    sum = np.zeros((p, p))
    upsilon = 0
    upsilon_2 = 0
    time_list=[]
    for i in range(K):
        time_list.append(res_list[i]['time'])
        x = data_env.data[res_list[i]['label_ids'], :p]
        sigma_p = np.linalg.inv(x.T.dot(x))
        sum += (res_list[i]['N'] / N) ** 2 * sigma_p
        mu = res_list[i]['N'] * np.linalg.inv(x.T.dot(x))
        mu_2 = lam_max(res_list[i]['N'] * np.linalg.inv(x.T.dot(x)))
        upsilon += res_list[i]['N'] / N * mu
        upsilon_2 += res_list[i]['N'] / N * mu_2
        beta_est += res_list[i]['N'] * res_list[i]['beta_est']
        label_ids += res_list[i]['label_ids']
    sigma_n = np.linalg.inv(sum)
    beta_est = beta_est / N
    S_n = (beta_est - beta0).T.dot(sigma_n).dot(beta_est - beta0)
    cp = int(S_n / N <= params['d'] ** 2 / upsilon_2)
    res_all = {}
    res_all['N'] = N
    res_all['beta_est'] = beta_est
    res_all['cp'] = cp
    res_all['method'] = res_list[0]['method']
    res_all['adaptive'] = res_list[0]['adaptive']
    res_all['time'] = np.max(time_list)
    return res_all


def get_info(params, data_env):
    return {'N': params['N'],
            'beta_est': params['beta_est'],
            'label_ids': data_env.labeled_ids,
            }

### important params setting
params0 = {
    'N': int(10000),  # total generated data size
    'p': int(5),      # length of beta
    'test_ratio': 0.3,   # test_ratio
    'simulations': int(100),  # simulation times
    'core': int(16),   # num of cores in parallel computing
    'd': 0.5, # same as d defination in paper
    'init_N': int(8), # initial sample size in each procedure
    'beta0': [-1, 1, 0.7, 0.5, 0.2], # real beta
    'note': 'dsep', # result saved directory
    'rho': 0, # correlated index
    'K': 5  # num of distributed sequential procedures
}


# def evaluation(data_env, params):
#     beta_est = params['beta_est']
#     p = beta_est.shape[0]
#     X = data_env.test[:, :p]
#     tmp = np.dot(X, beta_est)
#     Y_true = list(data_env.test[:, p])
#     accuracy = accuracy_score(Y_true, Y_pred)
#     precision = precision_score(Y_true, Y_pred)
#     recall = recall_score(Y_true, Y_pred)
#     F1 = f1_score(Y_true, Y_pred)
#     cm = confusion_matrix(Y_true, Y_pred)
#     confusion = cm.astype('float') / cm.sum(axis=1)
#     fpr, tpr, thresholds = roc_curve(Y_true, prob_list)
#     roc_auc = auc(fpr, tpr)
#     return {'RMSE': rmse}


def print_params(params):
    '''
    :param params:
    :return: save txtfile title
    '''
    path = './' + params['note']
    res_dir = path + '/rho' + str(params['rho']) + '/d_' + str(params['d'])
    isExists = os.path.exists(res_dir)
    if not isExists:
        os.makedirs(res_dir)
    date_time = datetime.now().strftime('%m-%d')
    file_name = res_dir + '/' + date_time + '.txt'
    f = open(file_name, 'a')
    f.write("==========================================\n")
    f.write("============== Results ===================\n")
    f.write("==========================================\n")
    params_print = {'initial sample size': params['init_N'],
                    'p': params['p'],
                    'd': params['d'],
                    'rho': params['rho'],
                    'beta0': params['beta0'],
                    'K': params['K']
                    }
    for key in params_print:
        f.write(key + ':' + str(params_print[key]) + '\n')
    start_time = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
    f.write('Start:' + start_time + '\n')
    params['res_dir'] = res_dir
    params['res_path'] = file_name
    f.close()


def save_tab(params, res_df):
    '''
    :param params:
    :param res_df:
    :return: save N C.P. time in files
    '''
    f = open(params['res_path'], 'a')
    end_time = datetime.now().strftime('%Y-%m-%d,%H:%M:%S')
    f.write('Repetition:' + str(params0['Repetition']) + '\n')
    f.write('End:' + end_time + '\n')
    f.write("---------------- END --------------------\n")
    f.write('Method' + '&' + '&N' + '&C.P' + '&Time' + '&beta' + '\\\\\n')
    beta_dict = {}
    for name in res_df.columns:
        print(name)
        df1 = pd.DataFrame(list(res_df[name]))
        adaptive = df1['adaptive'][0]
        method = df1['method'][0]
        beta_mean = np.mean(df1['beta_est'].values)
        beta_sd = np.sqrt(np.var(df1['beta_est'].values))
        beta_dict.update({name: np.vstack((beta_mean, beta_sd))})
        df1 = df1[['N', 'cp', 'time']]
        res_tab = df1.describe().loc[['mean', 'std']]
        for stat in ['mean', 'std']:
            f.write(method + '_' + adaptive + '&' + stat)
            for value in list(res_tab.loc[stat]):
                f.write('&' + str('%.3f' % value))
            f.write('\\\\\n')
    for name in res_df.columns:
        df1 = pd.DataFrame(list(res_df[name]))
        adaptive = df1['adaptive'][0]
        method = df1['method'][0]
        beta = beta_dict[name]
        f.write(method + '_' + adaptive + '&' + 'mean&' + '&'.join(str('%.3f' % i) for i in beta[0, :]))
        f.write('\\\\\n')
        f.write(method + '_' + adaptive + '&' + 'sd&' + '&'.join(str('%.3f' % i) for i in beta[1, :]))
        f.write('\\\\\n')
    f.close()


def simulation(params0):
    '''
    :param params0:
    :return: simulation result
    '''
    params = params0.copy()
    init_env = init_data(params)
    data_env = copy.deepcopy(init_env)
    LSER = distributed_seq_ana(data_env, params, adaptive='random', method='lse')
    params = params0.copy()
    data_env = copy.deepcopy(init_env)
    LSED = distributed_seq_ana(data_env, params, adaptive='D_optimal', method='lse')
    simu_res = {'LSER': LSER, 'LSED': LSED}
    simu_df = pd.DataFrame(simu_res)
    #res_list.append(simu_res)
    for methodName in simu_df.columns:
        fileName = params['res_dir'] + '/betaTab_d' + str(params['d']) + methodName + datetime.now().strftime(
            '%m_%d') + '.csv'
        with open(fileName, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(simu_df[methodName].loc['beta_est'])
    return simu_res


if __name__ == '__main__':
    d_list = [0.5, 0.4, 0.3, 0.2]
    k_list = [1, 2, 5]
    for d in d_list:
        for k in k_list:
            params0['d'] = d
            params0['K'] = k
            print_params(params0)
            core = params0['core']
            pool = mp.Pool(core)
            start_t = datetime.now()
            results = [pool.apply_async(simulation, args=(params0,)) for j in range(params0['simulations'])]
            res_list = [p.get() for p in results]
            res_df = pd.DataFrame(list(filter(None, res_list)))
            dump(res_list, params0['res_dir'] + '/res_list.sav')
            dump(res_df, params0['res_dir'] + '/res_df.sav')
            dump(params0, params0['res_dir'] + '/params0.sav')
            params0['Repetition'] = len(res_df)
            save_tab(params0, res_df)
            end_t = datetime.now()
            elapsed_sec = (end_t - start_t).total_seconds()
            print("multi-tread: " + "{:.2f}".format(elapsed_sec / 60) + " minutes")
