import scipy.integrate as spi
import numpy as np
from ..config import parse_config

config = parse_config.read_config()
globals().update(config)
cur_stage = stage[csn]
label = cur_stage['label']
delay = cur_stage['latent_delay']
tr_ban = policy['tr_ban']
# print('split_index:', split_index)

rate = [0.02759622, 0.02774166, 0.02969392, 0.03198433, 0.03280033,
    0.03305785, 0.03607214, 0.0394453 , 0.03802771, 0.04226312,
    0.044709  , 0.0421758 , 0.04411765, 0.04345299, 0.04776471,
    0.04825654, 0.04863481, 0.04789978, 0.05276441, 0.05360206,
    0.05654319, 0.05900053, 0.06115577, 0.06518395, 0.06806335,
    0.07166403, 0.07568763, 0.08193953, 0.08719403, 0.0895784 ,
    0.10260375, 0.09670442, 0.10672868, 0.11505533, 0.12405707,
    0.13183176, 0.13257773, 0.14565696, 0.16041379, 0.17075603,
    0.17844346, 0.21074481, 0.23019969, 0.2389292 , 0.2194623 ,
    0.2397541 , 0.29415042, 0.17375612, 0.31122995, 0.24715447,
    0.25305739, 0.2492891 , 0.27672209, 0.27674024, 0.47606383,
    0.38809035, 0.47983871, 0.46078431, 0.38693467, 0.57317073,
    0.44274809, 0.37864078, 0.43298969, 0.53658537, 0.5106383 ,
    0.5       , 0.57142857, 0.57142857, 0.66666667]

tr_2019 = [3.7253, 3.5147, 4.3003, 4.4622, 4.2665, 4.3427, 4.419 , 3.702 ,
       3.45  , 4.1991, 4.1418, 4.1328, 4.2874, 4.152 , 3.488 , 3.2698,
       3.9469, 3.8103, 3.5881, 3.5472, 3.0917, 2.845 , 2.3166, 1.9864,
       2.031 , 2.0023, 1.894 , 1.9139, 1.9893, 2.1716, 2.9441, 3.3213,
       3.5623, 3.6567, 3.64  , 3.1617, 3.0862, 3.9827, 3.8194, 4.0143,
       4.0514, 4.2747, 3.6909, 3.5353, 4.3351, 4.2806, 4.3835, 4.6221,
       4.6993, 3.9236, 3.8279, 4.4992, 4.5464, 4.4996, 4.5051, 4.7287,
       4.0333, 3.9043, 4.5879, 4.6274, 4.6469, 4.5595, 4.8362, 4.1974,
       3.8384, 4.5331, 4.5937, 4.6194, 4.5498, 4.7073, 4.0877, 3.8641,
       4.6319, 4.5567, 4.4931, 4.5133]

tr_2020 = [4.2826, 5.2757, 5.334 , 4.3842, 4.1317, 5.2662, 5.2815, 5.3214,
    5.1513, 5.2171, 4.4613, 4.2045, 5.1656, 5.1121, 4.9132, 4.9241,
    4.8733, 4.0436, 4.3249, 4.2152, 3.5992, 2.8838,
    1.9621, 1.2811, 0.8931, 0.6259, 0.6573, 0.6747, 0.6829, 0.6645, 0.6906,
    0.6902, 0.6617, 0.693 , 0.6791, 0.6667, 0.6   , 0.6171, 0.609 ,
    0.616 , 0.6168, 0.6076, 0.5958, 0.6191, 0.6714, 0.5963, 0.5791,
    0.5928, 0.5785, 0.5739, 0.5907, 0.5968, 0.5687, 0.5693, 0.6197,
    0.6107, 0.6198, 0.6085, 0.613 , 0.6184, 0.6157, 0.654 , 0.6523,
    0.6699, 0.6743, 0.677 , 0.6507, 0.65  , 0.70,   0.72,   0.73,
    0.73,   0.75,   0.71,   0.71,   0.81]

if(tr_ban):
    tr = tr_2020
else:
    tr = tr_2019

def get_tr(t):
    if(label == 1):
        idx = int(t)
        # tr = [4.2826, 5.2757, 5.334 , 4.3842, 4.1317, 5.2662, 5.2815, 5.3214,
        #     5.1513, 5.2171, 4.4613, 4.2045, 5.1656, 5.1121, 4.9132, 4.9241,
        #     4.8733, 4.0436, 4.3249, 4.2152, 3.5992, 2.8838]
    else:
        idx = int(t) + split_index
        # tr = [1.9621, 1.2811, 0.8931, 0.6259, 0.6573, 0.6747, 0.6829, 0.6645, 0.6906,
        #     0.6902, 0.6617, 0.693 , 0.6791, 0.6667, 0.6   , 0.6171, 0.609 ,
        #     0.616 , 0.6168, 0.6076, 0.5958, 0.6191, 0.6714, 0.5963, 0.5791,
        #     0.5928, 0.5785, 0.5739, 0.5907, 0.5968, 0.5687, 0.5693, 0.6197,
        #     0.6107, 0.6198, 0.6085, 0.613 , 0.6184, 0.6157, 0.654 , 0.6523,
        #     0.6699, 0.6743, 0.677 , 0.6507, 0.65  , 0.70,   0.72,   0.73,
        #     0.73,   0.75,   0.71,   0.71,   0.81]
    # if(t < len(tr)):
    #     return tr[int(t)]
    # else:
    #     return 1.0
    if(idx >= len(tr)):
        return 0.7
    return tr[idx]

def get_u_confirm_rate(t):
    if(label == 1):
        idx = int(t)
        # rate = [0.02759622, 0.02774166, 0.02969392, 0.03198433, 0.03280033,
        #     0.03305785, 0.03607214, 0.0394453 , 0.03802771, 0.04226312,
        #     0.044709  , 0.0421758 , 0.04411765, 0.04345299, 0.04776471,
        #     0.04825654, 0.04863481, 0.04789978, 0.05276441, 0.05360206,
        #     0.05654319, 0.05900053]
    else:
        # return 0.04216
        idx = int(t) + split_index
        # rate = [0.06115577, 0.06518395, 0.06806335,
        #     0.07166403, 0.07568763, 0.08193953, 0.08719403, 0.0895784 ,
        #     0.10260375, 0.09670442, 0.10672868, 0.11505533, 0.12405707,
        #     0.13183176, 0.13257773, 0.14565696, 0.16041379, 0.17075603,
        #     0.17844346, 0.21074481, 0.23019969, 0.2389292 , 0.2194623 ,
        #     0.2397541 , 0.29415042, 0.17375612, 0.31122995, 0.24715447,
        #     0.25305739, 0.2492891 , 0.27672209, 0.27674024, 0.47606383,
        #     0.38809035, 0.47983871, 0.46078431, 0.38693467, 0.57317073,
        #     0.44274809, 0.37864078, 0.43298969, 0.53658537, 0.5106383 ,
        #     0.5       , 0.57142857, 0.57142857, 0.66666667]
    if(idx >= len(rate)):
        return 0.6
    return rate[idx]

def SUIR(INPUT, t, beta, sigma, pho, epsilon, _lambda_, gamma):
    Y = np.zeros(len(INPUT))
    
    beta = max(beta, 0) 
    sigma = beta
    if(delay == 3):
        sigma_tr = 0.4847
        # sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.5474) * 1.2027
        sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.5474)
        # sigma_tr = np.exp(sigma_tr * get_tr(t))
    elif(delay == 5):
        sigma_tr = 0.5314
        # sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.7621) * 1.2346
        sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.7621)
        # sigma_tr = np.exp(sigma_tr * get_tr(t))
    elif(delay == 7):
        sigma_tr = 0.6171
        # sigma_tr = np.exp(sigma_tr * get_tr(t) - 3.0087) * 1.2690
        sigma_tr = np.exp(sigma_tr * get_tr(t) - 3.0087)
        # sigma_tr = np.exp(sigma_tr * get_tr(t))
    sigma = sigma * sigma_tr
    sigma = min(sigma, 1)  

    pho = max(pho, 0)

    epsilon = get_u_confirm_rate(t)

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

def SEUIR(INPUT, t, beta, sigma, pho, alpha, _lambda_, gamma):
    Y = np.zeros(len(INPUT))
    
    beta = max(beta, 0) 
    drop = sigma
    if(delay == 3):
        sigma_tr = 0.4847
        # sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.5474) * 1.2027
        sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.5474)
        # sigma_tr = np.exp(sigma_tr * get_tr(t))
    elif(delay == 5):
        # sigma_tr = 0.5314
        sigma_tr = 0.5738 # incubation 6.41
        # sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.7621) * 1.2346
        # sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.7621)
        sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.8489)
        # sigma_tr = np.exp(sigma_tr * 5 * (1 - drop) - 2.8489)
        # sigma_tr = np.exp(sigma_tr * get_tr(t))
    elif(delay == 7):
        sigma_tr = 0.6171
        # sigma_tr = np.exp(sigma_tr * get_tr(t) - 3.0087) * 1.2690
        sigma_tr = np.exp(sigma_tr * get_tr(t) - 3.0087)
        # sigma_tr = np.exp(sigma_tr * get_tr(t))
    sigma = beta
    sigma = sigma * sigma_tr
    sigma = min(sigma, 1)  
    
    # if(delay == 3):
    #     if(label == 1):
    #         pho = 0.014275        
    #     else:
    #         pho = 0.108717
    # elif(delay == 5):
    #     if(label == 1):
    #         pho = 0.027698        
    #     else:
    #         pho = 0.110018
    # elif(delay == 7):
    #     if(label == 1):
    #         pho = 0.042296        
    #     else:
    #         pho = 0.109755
    # else: # no delay in split
    
    # if(label == 1):
    #     pho = 0.002956        
    # else:
    #     pho = 0.103893
    
    # pho = pho * 5

    alpha = max(alpha, 0)
    alpha = min(alpha, 1)
    
    # epsilon = max(epsilon, 0)
    epsilon = get_u_confirm_rate(t)
    
    _lambda_ = max(_lambda_, 0)
    _lambda_ = min(_lambda_, 1)
    
    gamma = max(gamma, 0)
    
    S = INPUT[0]
    E = INPUT[1]
    U = INPUT[2]
    I = INPUT[3]
    IS = INPUT[4]
    R = INPUT[5]
    Y[0] = - (beta * S * I) - (sigma * S * max((U - (pho * IS)), 0.005*U))
    Y[1] = (beta * S * I) + (sigma * S * max((U - (pho * IS)), 0.005*U)) - (alpha * E)
    Y[2] = (alpha * E) - (epsilon * U)
    Y[3] = ((1 - _lambda_) * (epsilon * U)) - (gamma * I)
    Y[4] = (_lambda_ * (epsilon * U)) - (gamma * IS)
    Y[5] = (gamma * (I + IS))

    return Y

# def SEUIR(INPUT, t, beta, sigma, pho, alpha, _lambda_, gamma):
#     Y = np.zeros(len(INPUT))
    
#     beta = max(beta, 0) 
#     sigma = beta
#     if(delay == 3):
#         sigma_tr = 0.4847
#         # sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.5474) * 1.2027
#         sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.5474)
#         # sigma_tr = np.exp(sigma_tr * get_tr(t))
#     elif(delay == 5):
#         sigma_tr = 0.5314
#         # sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.7621) * 1.2346
#         sigma_tr = np.exp(sigma_tr * get_tr(t) - 2.7621)
#         # sigma_tr = np.exp(sigma_tr * get_tr(t))
#     elif(delay == 7):
#         sigma_tr = 0.6171
#         # sigma_tr = np.exp(sigma_tr * get_tr(t) - 3.0087) * 1.2690
#         sigma_tr = np.exp(sigma_tr * get_tr(t) - 3.0087)
#         # sigma_tr = np.exp(sigma_tr * get_tr(t))
#     sigma = sigma * sigma_tr
#     sigma = min(sigma, 1)  

#     pho = max(pho, 0)

#     alpha = 1 / delay
    
#     epsilon = get_u_confirm_rate(t)
    
#     _lambda_ = max(_lambda_, 0)
#     _lambda_ = min(_lambda_, 1)
    
#     gamma = max(gamma, 0)
    
#     S = INPUT[0]
#     E = INPUT[1]
#     U = INPUT[2]
#     I = INPUT[3]
#     IS = INPUT[4]
#     R = INPUT[5]
#     Y[0] = - (beta * S * I) - (sigma * S * max((U - (pho * IS)), 0.005*U))
#     Y[1] = (beta * S * I) + (sigma * S * max((U - (pho * IS)), 0.005*U)) - (alpha * E)
#     Y[2] = (alpha * E) - (epsilon * U)
#     Y[3] = ((1 - _lambda_) * (epsilon * U)) - (gamma * I)
#     Y[4] = (_lambda_ * (epsilon * U)) - (gamma * IS)
#     Y[5] = (gamma * (I + IS))

#     return Y

def SEIISR_v_base_eqs(INPUT, t, beta, sigma, pho, epsilon, _lambda_, gamma):
    if(model_name == 'SUIR'):
        return SUIR(INPUT, t, beta, sigma, pho, epsilon, _lambda_, gamma)
    else:
        return SEUIR(INPUT, t, beta, sigma, pho, epsilon, _lambda_, gamma)
