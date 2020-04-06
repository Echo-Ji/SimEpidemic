import os
import numpy as np 
import pandas as pd
import datetime as dt
import scipy.integrate as spi
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.sans-serif'] = 'NSimSun,Times New Roman' # 中文设置成宋体，除此之外的字体设置成New Roman  
# myfont = FontProperties(fname='./objs/simhei.ttf', size=14)

from src.utils import *
from src.data import data_loader
from src.model import model
from src.model import loss
from src.config import parse_config

config = parse_config.read_config()
globals().update(config)

def plot_case(PRED, TRUE, date_start, date_end, cityname='wuhan', save=True):
    disp_days = len(PRED) # len(TRUE)
    base_time = str_to_dt(date_start)
    t_range_subdt = [base_time + dt.timedelta(days = x) for x in range(len(PRED))]
    
    plt.figure(figsize=(8,6))
    plt.plot(t_range_subdt[:disp_days], PRED[:disp_days], 'b+-')
    plt.plot(t_range_subdt[:len(TRUE)], TRUE, "k*:")
    plt.grid("True")
    plt.legend(["Model Prediction", "Official Announcement"])
    # plt.title(u'{} 累计确诊数预测结果 (数据截止 {})'.format(cityname, alldates[-1]), FontProperties=myfont)
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.gcf().autofmt_xdate()
    if(save):
        plt.savefig(os.path.join(FIG_ROOT, '{0}_accum_infected_{1}_{2}.png'.format(cityname, date_start, date_end)), dpi=200)
    plt.show()
    return 

def bestParam(init_S, infect_data, remove_data, u_data, N, E0=None):
    # [1e-08, 7e-09, 15, 0.95, 0.98, 0.2]
    param_grid = {
        'beta': np.arange(3e-9, 3e-8, 2e-9),
        'sigma': np.arange(3e-9, 2e-8, 2e-9),
        'pho': np.arange(1, 18, 2),
        'epsilon': np.arange(0.2, 1.01, 0.1),
        '_lambda_': np.arange(0.8, 1.01, 0.02), 
        'gamma': np.square(np.arange(1e-3, 1, 0.05)), 
        }
    param_name = ['beta', 'sigma', 'pho', 'epsilon', '_lambda_', 'gamma']
    S0 = init_S
    I0 = infect_data[0] - remove_data[0]
    if (I0 == 0):
        I0 = 1.
    E0 = infect_data[0] * 3. if E0 is None else E0
    R0 = remove_data[0]
    INPUT = (S0, E0, 0.0, I0, R0)
    t_range = np.arange(0.0, len(infect_data), 1.0)
    # TRUE = np.random.rand(len(infect_data), 3) * 1e6
    # TRUE[:, 2] = infect_data
    TRUE = np.stack((infect_data, remove_data, u_data), axis=-1)
    ALL_RES, minind = loss.GridSearch(model.SEIISR_v_base_eqs, param_grid, param_name, INPUT, t_range, TRUE, loss.SEIISR_loss)
    best_x = tuple([ALL_RES[minind]['parameters'][name] for name in param_name]) 
    return best_x
    # t_range = np.arange(0.0, N, 1.0)
    # PRED = spi.odeint(SEIR_base_eqs, INPUT, t_range, )
    # return PRED[:, 2], {'S': PRED[:, 0], 'E': PRED[:, 1], 'I': PRED[:, 2], 'R': PRED[:, 3], 'I+R': PRED[:, 2] + PRED[:, 3], 'ACTUAL_ALL': PRED[:, 1] + PRED[:, 2] + PRED[:, 3]}

def pred(init_S, TRUE, x0_init, label, optimize=True, pred_length=0):
    print()
    print('='*20)
    print('Simulating the epidemic...')
    S0 = init_S
    I0, R0, U0 = TRUE[0]
    I0 = I0 - R0
    if(I0 == 0.):
        I0 = 1
    INPUT = (S0, U0, 0.0, I0, R0)
    print(INPUT)

    sim_length = len(TRUE[:, 0])
    t_range = np.arange(0, sim_length, 1)
    x0 = x0_init
    r = loss.optim_fun((INPUT, t_range, TRUE))(x0)
    print('loss before:', r)
    
    if(optimize):
        RES = minimize(loss.optim_fun((INPUT, t_range, TRUE)), x0, method = 'Nelder-Mead', options = {'disp': True})
        x = RES.x
    else:
        if(label == 3):
            # x = np.array([-3.54890159e-10, 3.26319271e-06, 1.38017850e+02, 2.51030231e-01, -3.69255824e+01, 3.44165509e-02])
            x[1] = x[1]*0.902
        elif(label == 1):
            # x = np.array([3.99e-08, 7.52e-09, 31.3, 3.34e-01, 1.92e-01, 1.19e-03]) # 8.3713
            # x = np.array([1.01e-07, 2.15e-09, 11.8, 5.68e-02, 3.17e-01, 5.49e-03]) # 11.212 initial
            # x = np.array([1.01e-07, 2.24e-09, 11.8, 3.85e-02, 2.66e-01, 5.49e-03]) # 11.212 modified
            # x = np.array([4.912e-08, 7.039e-09, 1.79, 2.83e-02, 3.24e-01, 1.33e-02]) # infection delay
            x = np.array([1.04e-08, 1.04e-08, 1.287, 3.60e-02, 5.79e-01, 7.94e-03])
        elif(label == 2):
            # x = np.array([1.4e-09, 2.5e-08, 2.73, 5.55e-02, 3.73e-01, 4.49e-02]) # 8.37
            # x = np.array([7.80e-08, 1.52e-08, 1.69, 3.85e-02, 9.96e-01, 6.37e-02]) # 11.212 initial
            # x = np.array([7.80e-08, 1.53e-08, 1.69, 3.85e-02, 9.98e-01, 6.37e-02]) # 11.212 modified
            # x = np.array([4.912e-08, 7.039e-09, 1.79, 2.83e-02, 9.99e-01, 4.37e-02]) # infection delay
            x = np.array([1.04e-08, 1.04e-08, 1.287, 3.60e-02, 1, 3.44e-02])
        
    r = loss.optim_fun((INPUT, t_range, TRUE))(x)
    print('loss after:', r)
    print('best x:', x)
    
    RES = spi.odeint(model.SEIISR_v_base_eqs, (INPUT[0], INPUT[1], INPUT[2], INPUT[3], INPUT[4]), t_range, args=(x[0], x[1], x[2], x[3], x[4], x[5]))
    print()
    print('='*20)
    print('Simulation Result:\n', RES)
    if(pred_length > 0):
        RES0 = RES
        # other_U = max(0, RES0[-1, 2] + RES0[-1, 3] + RES0[-1, 4] - TRUE[-1, 0])
    
        RES0[-1, 4] = TRUE[-1, 1]
        RES0[-1, 3] = (TRUE[-1, 0] - TRUE[-1, 1])
        RES0[-1, 2] = 0.
        # RES0[-1, 1] += other_U
        RES0[-1, 1] = TRUE[-1, 2]

        t_range = np.arange(0, sim_length+pred_length, 1)
        RES1 = spi.odeint(model.SEIISR_v_base_eqs, RES0[-1, :], t_range[sim_length-1:], args = (x[0], x[1], x[2], x[3], x[4], x[5]))
        RES = np.concatenate((RES0[:-1, :], RES1))
    return RES

def save(file_name, cases, start_date):
    base_time = str_to_dt(start_date)
    date_range = [base_time + dt.timedelta(days = x) for x in range(len(cases))]
    
    df = pd.DataFrame({'date': date_range, 'value': cases})
    # print(df.head())
    df.to_csv(file_name, index=False)
    return 

def sim(cur_stage=stage1, save_pred=False, find_best=False):
    print()
    print('='*20)
    # parameter setting
    S0 = 11.212e6
    start_date = cur_stage['start_date']
    end_date = cur_stage['end_date']
    label = cur_stage['label']
    print('city: {0}, \npopulation: {1}, \nstart date: {2}, \nend date: {3}'.format(city, S0, start_date, end_date))

    # data loading
    I = data_loader.load_I(start_date=start_date, end_date=end_date)
    R = data_loader.load_R(start_date=start_date, end_date=end_date)
    U = data_loader.load_U(start_date=start_date, end_date=end_date)
    TRUE = np.stack((I, R, U), axis=-1)
    print('Time span (days): ', len(TRUE[:, 0]))
    print('True data:\n', TRUE)
    
    if(find_best):
        # find best paras
        print('Grid-Searching for the best parameters...')
        best_x = bestParam(S0, I, R, U, len(I), U[0])
        print(best_x)
    else:
        # model simulating
        if(label == 1):
            x0 = np.array([5e-8, 1e-8, 5, 0.3, 0.95, 9e-3])
            # x0 = np.array([5e-08, 2e-08, 20, 6.6e-01, 0.9, 0.06]) 
        elif(label == 2):
            # (2.9e-08, 1.9e-08, 1, 0.2, 0.8, 0.0228)  ub_higher: (2e-08, 6.9e-09, 15, 0.95, 0.98, 0.2) best: 1e-08, 6.9e-09, 15, 0.95, 0.98, 0.2
            # x0 = np.array([1e-08, 6.9e-09, 15, 0.95, 0.98, 0.2]) # pho 调节上限  without tr
            # x0 = np.array([2.98e-08, 1.78e-08, 3.09e-01, 1.01e-01, 1, 2.82e-02])
            x0 = np.array([2.9e-08, 1.9e-08, 1, 0.95, 0.98, 0.138])
        elif(label == 3):
            # x0 = np.array([1.2e-08, 1.85e-08, 3, 0.98, 0.999, 0.18]) # [2e-08, 2.25e-07, 3, 0.98, 0.999, 0.08] earier
            # x0 = np.array([1e-08, 1.8e-07, 0.8, 0.98, 0.999, 0.18])
            x0 = np.array([2e-08, 2.25e-07, 3, 0.98, 0.999, 0.08])
        else:
            # x0 = np.array([5e-8, 1e-8, 5, 0.3, 0.95, 9e-3])
            x0 = np.array([5e-08, 2e-08, 20, 6.6e-01, 0.9, 0.06]) 
            # [ 1.76076889e-08  1.16861794e-09  5.85497051e+01  2.36116203e+00
#  -8.05737153e-01  6.82714778e-02]
            # 5.78388906e-08, 2.97272370e-08, 5.88479924e+00, 8.86293340e-02,  8.56928458e-01, 2.52213892e-02
            # # 2.9e-08, 1.9e-08, 1, 1, 0.8, 1e-06
        # RES = pred(S0, TRUE, x0, pred_length=70)
        RES = pred(S0, TRUE, x0, label, optimize=False)
        
        # result ploting
        PRED_ALL = RES[:, 1] + RES[:, 2] + RES[:, 3] + RES[:, 4] # 发病 = U + I + IS + R
        TRUE_ALL = U + I # 
        plot_case(PRED=PRED_ALL, TRUE=TRUE_ALL, date_start=start_date, date_end=end_date, cityname='0405-wuhan-trs', save=True)
        if(save_pred):
            save(os.path.join(DATA_ROOT, '{0}.csv'.format(start_date)), PRED_ALL, start_date)
    return 

def merge_draw():
    cur_stage = stage_all
    start_date = cur_stage['start_date']
    end_date = cur_stage['end_date']

    # data loading
    I = data_loader.load_I(start_date=start_date, end_date=end_date)
    R = data_loader.load_R(start_date=start_date, end_date=end_date)
    # print(len(I), len(R))
    U = data_loader.load_U(start_date=start_date, end_date=end_date)
    TRUE = np.stack((I, R, U), axis=-1)
    print(len(TRUE[:, 0]), TRUE)

    PRED_ALL = pd.read_csv(os.path.join(DATA_ROOT, 'pred_tr.csv'), index_col=0)['value'].values
    TRUE_ALL = U + I # 
    plot_case(PRED=PRED_ALL, TRUE=TRUE_ALL, date_start=start_date, date_end=end_date, cityname='wuhan-trs', save=False)


if __name__ == "__main__":
    sim(cur_stage=stage2, save_pred=0, find_best=0)
    # merge_draw()
    pass