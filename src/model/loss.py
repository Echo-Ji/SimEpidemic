import numpy as np
import scipy.integrate as spi
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid

from .model import SEIISR_v_base_eqs
from ..config import parse_config

config = parse_config.read_config()
globals().update(config)

def SEIISR_loss(TRUE, PRED):
    # print(TRUE.shape, PRED.shape)
    # loss = np.log(np.cosh())
    # return np.sum(np.square((TRUE[:, 0] - TRUE[:, 1]) - (PRED[:, 3] + PRED[:, 4]))) + np.sum(np.square(TRUE[:, 1] - PRED[:, 5])) + np.sum(np.square(TRUE[:, 2] - PRED[:, 2]))
    # return np.sum(np.square(TRUE[:, 3] - PRED[:, 1])) 
    # + np.sum(np.square((TRUE[:, 0] - TRUE[:, 1]) - (PRED[:, 3] + PRED[:, 4]))) + np.sum(np.square(TRUE[:, 1] - PRED[:, 5]))
    if(model_name == 'SUIR'):
        return np.sum(np.square((TRUE[:, 0] + TRUE[:, 2]) - (PRED[:, 1] + PRED[:, 2] + PRED[:, 3] + PRED[:, 4]))) + np.sum(np.square(TRUE[:, 1] - PRED[:, 4])) 
    else:
        return np.sum(np.square((TRUE[:, 0] + TRUE[:, 2]) - (PRED[:, 2] + PRED[:, 3] + PRED[:, 4] + PRED[:, 5]))) + np.sum(np.square(TRUE[:, 1] - PRED[:, 5]))
    # + 0.5 * np.sum(np.square(TRUE[:, 3] - PRED[:, 1]))
    # return np.sum(np.log1p(np.abs((TRUE[:, 0] + TRUE[:, 2]) - (PRED[:, 1] + PRED[:, 2] + PRED[:, 3] + PRED[:, 4])))) + np.sum(np.log1p(np.abs(TRUE[:, 1] - PRED[:, 4])))
    # return np.sum(np.log(np.cosh((TRUE[:, 0] - TRUE[:, 1]) - (PRED[:, 2] + PRED[:, 3])))) + np.sum(np.log(np.cosh(TRUE[:, 1] - PRED[:, 4]))) + np.sum(np.log(np.cosh(TRUE[:, 2] - PRED[:, 1])))
    # return np.sum(np.square((TRUE[:, 0] - TRUE[:, 2]) - (PRED[:, 1] + PRED[:, 2] + PRED[:, 3] + PRED[:, 4])))

def optim_fun(args):
    INPUT, t_range, TRUE = args
    '''
    x[0] = E[0]
    x[1] = beta
    x[2] = sigma
    x[3] = epsilon
    x[4] = _lambda_
    x[5] = gamma
    '''
    v = lambda x: SEIISR_loss(TRUE,
                              spi.odeint(SEIISR_v_base_eqs, INPUT, t_range, args=(x[0], x[1], x[2], x[3], x[4], x[5])))

    return v

def optim_bnd(N):
    bounds = [(0, None) for i in range(N)]

def optim_con(N):
    con = [{'type': 'ineq', 'fun': lambda x: x[i]} for i in range(N)]

    return con

def GridSearch(ode_eqs, param_grid, param_name, INPUT, t_range, TRUE, loss_func):
    ALL_RES = []
    Grid = ParameterGrid(param_grid)
    for grid in Grid:
        args = [grid[name] for name in param_name]
        PRED = spi.odeint(ode_eqs, INPUT, t_range, args=tuple(args))
        loss = loss_func(TRUE, PRED)
        ALL_RES.append({'loss': loss, 'prediction': PRED, 'parameters': grid})
        
    minloss = ALL_RES[0]['loss']
    minind = 0
    for i, res in enumerate(ALL_RES):
        if res['loss'] < minloss:
            minloss = res['loss']
            minind = i

    return ALL_RES, minind

if __name__ == '__main__':
    S0 = 1.3e8
    I0 = 41 - 14
    R0 = 14
    INPUT = (S0, I0 * 20, 0.0, I0, R0)

    #TRUE = np.array([[45, 62, 121, 199, 291, 440, 571, 830, 1287, 1975, 2744, 4515, 5974, 7711, 9731],
    #                 [15, 19, 24, 28, 31, 37, 45, 53, 74, 100, 126, 158, 237, 305, 389]])
    TRUE = np.array([[440, 571, 830, 1287, 1975, 2744, 4515, 5974, 7711, 9731, 11791],
                     [37, 45, 53, 74, 100, 126, 158, 237, 305, 389, 504]])
    TRUE = TRUE.T
    INPUT = (S0, INPUT[1], 0.0, 291 - 31, 31)

    t_range = np.arange(0, len(TRUE), 1)
    '''
    x0 = np.array([INPUT[1], 3e-9, 3e-9, 0.48, 0.0, 9e-3])
    #x0 = np.array([INPUT[1], 1e-9, 0.0, 0.1, 0.0, 1e-3])
    r = optim_fun((INPUT, t_range, TRUE))(x0)
    print(r)
    RES = minimize(optim_fun((INPUT, t_range, TRUE)), x0, method = 'Nelder-Mead', options = {'disp': True})

    #print(RES.success)
    print(RES.x)

    r = optim_fun((INPUT, t_range, TRUE))(RES.x)
    print(r)

    x = RES.x
    t_range = np.arange(0, 100)
    RES = spi.odeint(SEIISR_ode.SEIISR_base_eqs, (INPUT[0], x[0], INPUT[2], INPUT[3], INPUT[4]), t_range, args = (x[1], x[2], x[3], x[4], x[5]))
    '''

    x0 = np.array([5e-9, 7e-9, 1, 0.3, 0.97, 9e-3])
    #x0 = np.array([INPUT[1], 1e-9, 0.0, 0.1, 0.0, 1e-3])
    r = optim_fun((INPUT, t_range, TRUE))(x0)
    print('r:', r)
    RES = minimize(optim_fun((INPUT, t_range, TRUE)), x0, method = 'Nelder-Mead', options = {'disp': True})

    #print(RES.success)
    print('RES.x:', RES.x)

    r = optim_fun((INPUT, t_range, TRUE))(RES.x)
    print('r:', r)

    x = RES.x
    t_range = np.arange(0, 100)
    RES = spi.odeint(SEIISR_v_base_eqs, (INPUT[0], INPUT[1], INPUT[2], INPUT[3], INPUT[4]), t_range, args = (x[0], x[1], x[2], x[3], x[4], x[5]))

    print('RES:', RES)
