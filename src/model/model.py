import scipy.integrate as spi
import numpy as np

def SEIISR_base_eqs(INPUT, t, beta, sigma, epsilon, _lambda_, gamma):
    Y = np.zeros((5))
    #_lambda_ = np.exp((_lambda_ * (t - 6)))
    #_lambda = (_lambda_ / (1 + _lambda_))
    #print(_lambda_)
    beta = max(beta, 0)
    sigma = max(sigma, 0)
    epsilon = max(epsilon, 0)
    _lambda_ = max(_lambda_, 0)
    gamma = max(gamma, 0)
    Y[0] = - (beta * INPUT[0] * INPUT[2]) - (sigma * INPUT[0] * INPUT[1])
    Y[1] = (beta * INPUT[0] * INPUT[2]) + (sigma * INPUT[0] * INPUT[1])  - (epsilon * INPUT[1])
    Y[2] = ((1 - _lambda_) * (epsilon * INPUT[1])) - (gamma * INPUT[2])
    Y[3] = (_lambda_ * epsilon * INPUT[1]) - (gamma * INPUT[3])
    Y[4] = (gamma * INPUT[2]) + (gamma * INPUT[3])

    return Y

def get_tr(t):
    # print(t)
    # tr = [4.2826, 5.2757, 5.334 , 4.3842, 4.1317, 5.2662, 5.2815, 5.3214,
    #    5.1513, 5.2171, 4.4613, 4.2045, 5.1656, 5.1121, 4.9132, 4.9241,
    #    4.8733, 4.0436, 4.3249, 4.2152, 3.5992, 2.8838, 1.9621, 
    #    1.2811, 0.8931, 0.6259, 0.6573, 0.6747, 0.6829, 0.6645, 0.6906,
    #    0.6902, 0.6617, 0.693 , 0.6791, 0.6667, 0.6   , 0.6171, 0.609 ,
    #    0.616 , 0.6168, 0.6076, 0.5958, 0.6191, 0.6714, 0.5963, 0.5791,
    #    0.5928, 0.5785, 0.5739, 0.5907, 0.5968, 0.5687, 0.5693, 0.6197,
    #    0.6107, 0.6198, 0.6085, 0.613 , 0.6184, 0.6157, 0.654 , 0.6523,
    #    0.6699, 0.6743, 0.677 , 0.6507, 0.65  , 0.70]
    # tr = [4.2826, 5.2757, 5.334 , 4.3842, 4.1317, 5.2662, 5.2815, 5.3214,
    #    5.1513, 5.2171, 4.4613, 4.2045, 5.1656, 5.1121, 4.9132, 4.9241,
    #    4.8733, 4.0436, 4.3249, 4.2152, 3.5992, 2.8838]
    tr = [1.9621, 1.2811, 0.8931, 0.6259, 0.6573, 0.6747, 0.6829, 0.6645, 0.6906,
       0.6902, 0.6617, 0.693 , 0.6791, 0.6667, 0.6   , 0.6171, 0.609 ,
       0.616 , 0.6168, 0.6076, 0.5958, 0.6191, 0.6714, 0.5963, 0.5791,
       0.5928, 0.5785, 0.5739, 0.5907, 0.5968, 0.5687, 0.5693, 0.6197,
       0.6107, 0.6198, 0.6085, 0.613 , 0.6184, 0.6157, 0.654 , 0.6523,
       0.6699, 0.6743, 0.677 , 0.6507, 0.65  , 0.70]
    # tr = [1.2811, 0.8931, 0.6259, 0.6573, 0.6747, 0.6829, 0.6645, 0.6906,
    #    0.6902, 0.6617, 0.693 , 0.6791, 0.6667, 0.6   , 0.6171, 0.609]
    # tr = [0.616 , 0.6168, 0.6076, 0.5958, 0.6191, 0.6714, 0.5963, 0.5791,
    #    0.5928, 0.5785, 0.5739, 0.5907, 0.5968, 0.5687, 0.5693, 0.6197,
    #    0.6107, 0.6198, 0.6085, 0.613 , 0.6184, 0.6157, 0.654 , 0.6523,
    #    0.6699, 0.6743, 0.677 , 0.6507, 0.65  , 0.70]
    if(t < len(tr)):
        return np.log1p(tr[int(t)])
    else:
        return 1.0

def SEIISR_v_base_eqs(INPUT, t, beta, sigma, pho, epsilon, _lambda_, gamma):
    Y = np.zeros((5))
    beta = max(beta, 0) #* get_tr(t)
    sigma = max(sigma, 0) * get_tr(t) 
    pho = max(pho, 0)
    epsilon = max(epsilon, 0)
    _lambda_ = max(_lambda_, 0)
    _lambda_ = min(_lambda_, 1)
    gamma = max(gamma, 0)
    S = INPUT[0]
    E = INPUT[1] # U
    I = INPUT[2]
    IS = INPUT[3]
    R = INPUT[4]
    Y[0] = - (beta * S * I) - (sigma * S * max((E - (pho * IS)), 0.005*E))
    Y[1] = (beta * S * I) + (sigma * S * max((E - (pho * IS)), 0.005*E)) - (epsilon * E)
    Y[2] = ((1 - _lambda_) * (epsilon * E)) - (gamma * I)
    Y[3] = (_lambda_ * (epsilon * E)) - (gamma * IS)
    Y[4] = (gamma * (I + IS))

    return Y
