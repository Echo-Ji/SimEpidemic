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

# Parameters Settings
config = parse_config.read_config()
globals().update(config)

sim_flag = True
optimize = False
save_plot = False
save_pred = False

cur_stage = stage[csn]
label = cur_stage['label']
delay = cur_stage['latent_delay']
# policy parameters
tr_ban = policy['tr_ban']
jnt_is = policy['jnt_is']
hospital = policy['hospital']

today = dt.date.today()
fig_dir = os.path.join(FIG_ROOT, '{0}'.format(today))
if(os.path.exists(fig_dir) == False):
    os.mkdir(fig_dir)

def plot_case(PRED, TRUE, date_start, date_end, offset=7, cityname='wuhan', save=True):
    disp_days = len(PRED) # len(TRUE)
    base_time = str_to_dt(date_start)
    t_range_subdt = [base_time + dt.timedelta(days=x+offset) for x in range(len(PRED))]
    
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
        plt.savefig(os.path.join(fig_dir, '{0}_accfb_{1}_{2}.png'.format(cityname, date_start, date_end)), dpi=200)
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

def pred(init_S, TRUE, x0_init, label, delay, optimize=True, pred_length=0, drop=0.84, verbose=False):
    if(verbose):
        print()
        print('='*20)
        print('Simulating the epidemic...')

    S0 = init_S
    if(model_name == 'SUIR'):
        I0, R0, U0 = TRUE[0]
    else:
        I0, R0, U0, E0 = TRUE[0]
    I0 = I0 - R0
    S0 = S0 - np.sum(TRUE[0])
    if(I0 == 0.):
        I0 = 1
    if(model_name == 'SUIR'):
        INPUT = (S0, U0, 0.0, I0, R0)
    else:
        INPUT = (S0, E0, U0, 0.0, I0, R0)
    
    if(verbose):
        print(INPUT)

    sim_length = len(TRUE[:, 0])
    t_range = np.arange(0, sim_length, 1)
    x0 = x0_init
    r = loss.optim_fun((INPUT, t_range, TRUE))(x0)
    if(verbose):
        print('loss before:', r)
    
    if(optimize):
        RES = minimize(loss.optim_fun((INPUT, t_range, TRUE)), x0, method = 'Nelder-Mead', options = {'disp': verbose})
        x = RES.x
    else:
        pho = 3e-05 # joint isolation rate if no policy constrain
        _lambda = 0.0053 # isolation rate if no hospital is setup
        gamma = 6.33e-03 # removal rate if no hospital is setup
        if(label == 3):
            x[1] = x[1] * 0.902
        elif(label == 1):
            if(delay == 3):
                x = np.array([5.58e-08, -1, pho, 0.79e-01, 0.0055, 6.39e-03])
            elif(delay == 5):
                x = np.array([6.06e-08, -1, pho, 0.57e-01, _lambda, gamma])
            elif(delay == 7):
                x = np.array([6.45e-08, -1, pho, 0.50e-01, 0.0056, 6.23e-03])
        elif(label == 2):
            if(delay == 3):
                pho = 0.09 if jnt_is else pho
                x = np.array([5.59e-08, -1, pho, 1.92e-01, 1, 3.72e-02])
            elif(delay == 5):
                pho = 0.105 if jnt_is else pho
                _lambda = 1 if hospital else _lambda # isolation rate if no hospital is setup
                gamma = 3.69e-02 if hospital else gamma
                x = np.array([6.06e-08, drop, pho, 1.15e-01, _lambda, gamma])
            elif(delay == 7):
                pho = 0.15 if jnt_is else pho
                x = np.array([6.45e-08, -1, pho, 1.05e-1, 1, 3.70e-02])


    r = loss.optim_fun((INPUT, t_range, TRUE))(x)
    if(verbose):
        print('loss after:', r)
        print('best x:', x)
    
    RES = spi.odeint(model.SEIISR_v_base_eqs, INPUT, t_range, args=(x[0], x[1], x[2], x[3], x[4], x[5]))
    if(verbose):
        print()
        print('='*20)
        print('Simulation Result:\n', RES)

    if(pred_length > 0):
        RES0 = RES
        # other_U = max(0, RES0[-1, 2] + RES0[-1, 3] + RES0[-1, 4] - TRUE[-1, 0])
        
        # You should to set these values if the true future epidemic is expected to be predicted.
        # RES0[-1, 5] = TRUE[-1, 1]
        # RES0[-1, 4] = (TRUE[-1, 0] - TRUE[-1, 1])
        # RES0[-1, 3] = 0.
        # # RES0[-1, 1] += other_U
        # RES0[-1, 2] = TRUE[-1, 2]

        t_range = np.arange(0, sim_length+pred_length, 1)
        RES1 = spi.odeint(model.SEIISR_v_base_eqs, RES0[-1, :], t_range[sim_length-1:], args = (x[0], x[1], x[2], x[3], x[4], x[5]))
        RES = np.concatenate((RES0[:-1, :], RES1))
        if(verbose):
            print()
            print('='*20)
            print('Prediction Result:\n', RES1)
    return RES

def dec_days(base, data, ratio):
    ths = base * ratio
    ths = 1.0
    diff = np.diff(data)
    span = -1
    flag = 1
    while(True):
        span += 1
        if(span >= len(diff)):
            break

        if(diff[span] <= ths):
            print(' Days needed: {0}'.format(span + 1))
            flag = 0
            break
    if(flag):
        print(' Need more days...')
        return -1
    return span

def save(file_name, cases, start_date):
    base_time = str_to_dt(start_date)
    date_range = [base_time + dt.timedelta(days = x) for x in range(len(cases))]
    
    df = pd.DataFrame({'date': date_range, 'value': cases})
    # print(df.head())
    df.to_csv(file_name, index=False)
    return 

def sim(drop, pred_length=0, verbose=False, plot=True):
    print()
    print('='*20)
    # parameter setting
    S0 = 11.212e6
    start_date = cur_stage['start_date']
    end_date = cur_stage['end_date']
    if(verbose):
        print('city: {0}, \npopulation: {1}, \nstart date: {2}, \nend date: {3}'.format(city, S0, start_date, end_date))

    # data loading
    I = data_loader.load_I(start_date=start_date, end_date=end_date)
    R = data_loader.load_R(start_date=start_date, end_date=end_date)
    U = data_loader.load_U(start_date=start_date, end_date=end_date, offset=delay)
    E = data_loader.load_E(start_date=start_date, end_date=end_date, offset=delay)
    assert len(I) == len(E)
    if(model_name == 'SUIR'):
        TRUE = np.stack((I, R, U), axis=-1)
    else:
        TRUE = np.stack((I, R, U, E), axis=-1)

    if(verbose):
        print('Time span (days): ', len(TRUE[:, 0]))
        print('True data:\n', TRUE)
    
    find_best = 0
    if(find_best):
        # find best paras
        print('Grid-Searching for the best parameters...')
        best_x = bestParam(S0, I, R, U, len(I), U[0])
        print(best_x)
    else:
        # model simulating
        if(label == 1):
            x0 = np.array([5e-8, 1e-8, 5, 0.3, 0.95, 9e-3])
        elif(label == 2):
            x0 = np.array([2.9e-08, 1.9e-08, 1, 0.95, 0.98, 0.138])
        elif(label == 3):
            x0 = np.array([2e-08, 2.25e-07, 3, 0.98, 0.999, 0.08])
        else:
            x0 = np.array([5e-08, 2e-08, 20, 6.6e-01, 0.9, 0.06]) 
        # RES = pred(S0, TRUE, x0, pred_length=70)
        RES = pred(S0, TRUE, x0, label, delay, optimize=optimize, pred_length=pred_length, drop=drop/100, verbose=verbose)
        
        # result ploting
        use_E = False
        if(use_E):
            PRED_ALL = RES[:, 1]
            TRUE_ALL = E
            plot_offset = 7
        else:
            if(model_name == 'SUIR'):
                PRED_ALL = RES[:, 1] + RES[:, 2] + RES[:, 3] + RES[:, 4] # 发病 = U + I + IS + R
            else:
                PRED_ALL = RES[:, 2] + RES[:, 3] + RES[:, 4] + RES[:, 5] # 发病 = U + I + IS + R
            TRUE_ALL = U + I #
            plot_offset = 0

        today = dt.date.today()
        print(' Peak series: ', PRED_ALL[-5:])
        span = dec_days(1925, PRED_ALL, 0.5) # 23号新增：1925，ratio：0.5
        print(' Drop rate: {0}\n Peak value: {1}\n Diff peak: {2}'.format(drop, PRED_ALL[span], PRED_ALL[span] - PRED_ALL[0]))
        if(plot):
            plot_case(PRED=PRED_ALL, TRUE=TRUE_ALL, date_start=start_date, date_end=end_date, offset=plot_offset, 
                cityname='{0}-s{1}d{2}_t{3}j{4}h{5}'.format(today, label, delay, tr_ban, jnt_is, hospital), save=save_plot)
        if(save_pred):
            save(os.path.join(OUT_ROOT, 'pred_d{0}s{1}_t{2}j{3}h{4}.csv'.format(delay, csn, tr_ban, jnt_is, hospital)), PRED_ALL, start_date)
    return drop, span, PRED_ALL[span] - 1925

def merge_draw():
    print()
    print('='*20)
    start_date = cur_stage['start_date']
    end_date = cur_stage['end_date']
    print('city: {0}, \nstart date: {1}, \nend date: {2}, \ndelay days: {3}.'.format(city, start_date, end_date, delay))

    # data loading
    I = data_loader.load_I(start_date=start_date, end_date=end_date)
    R = data_loader.load_R(start_date=start_date, end_date=end_date)
    U = data_loader.load_U(start_date=start_date, end_date=end_date, offset=delay)
    E = data_loader.load_E(start_date=start_date, end_date=end_date, offset=delay)
    assert len(I) == len(E)
    # if(model_name == 'SUIR'):
    #     TRUE = np.stack((I, R, U), axis=-1)
    # else:
    #     TRUE = np.stack((I, R, U, E), axis=-1)
    # print(len(TRUE[:, 0]), TRUE)

    # PRED_ALL = pd.read_csv(os.path.join(OUT_ROOT, data_file), index_col=0)['value'].values
    PRED_ALL = []
    for i in range(1, 3, 1):
        data_file = 'pred_d{0}s{1}_t{2}j{3}h{4}.csv'.format(delay, i, tr_ban, jnt_is, hospital)
        PRED_ALL.extend(pd.read_csv(os.path.join(OUT_ROOT, data_file), index_col=0)['value'].values)
    TRUE_ALL = U + I # 
    
    plot_offset = 0
    plot_case(PRED=PRED_ALL, TRUE=TRUE_ALL, date_start=start_date, date_end=end_date, offset=plot_offset, 
        cityname='all_{0}-d{1}'.format(today, delay), save=save_plot)
    return 

if __name__ == "__main__":
    if(sim_flag):
        sim(drop=100, verbose=True)
        # res = []
        # for drop_rate in range(0, 101, 1):
        #     res.append(list(sim(drop_rate, pred_length=100, plot=False)))
        # res = np.array(res, dtype=np.int)
        # df = pd.DataFrame({'BMI drop (%)': res[:, 0], 'Peak days': res[:, 1], 'Peak diff': res[:, 2]})
        # df.to_csv(os.path.join(OUT_ROOT, 'loss.csv'), index=False)
        # print(res)
    else:
        merge_draw()
    pass