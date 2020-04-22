import os 
import numpy as np 
import pandas as pd 
from datetime import date
from datetime import timedelta
from ..config import parse_config

def load_R(city='武汉市', start_date='2020-01-18', end_date='2020-01-23'):
    config = parse_config.read_config()
    globals().update(config)

    file_name = 'epi_initial-67.xlsx'
    df = pd.read_excel(os.path.join(DATA_ROOT, file_name), sheet_name='城市死亡人数', index_col=0)
    df = df.fillna(0)
    city_death = df.set_index('city').loc[city]
    df = pd.read_excel(os.path.join(DATA_ROOT, file_name), sheet_name='城市出院人数', index_col=0)
    df = df.fillna(0)
    city_cure = df.set_index('city').loc[city]
    city_R = pd.DataFrame({'death': city_death[1:], 'cure': city_cure[1:]})
    city_R['R'] = city_R['death'] + city_R['cure']
    
    t1 = date.fromisoformat(start_date)
    t2 = date.fromisoformat(end_date)

    R = city_R.loc[t1:t2]['R'].values.astype(np.int)
    return R

def load_I(city='武汉市', start_date='2020-01-18', end_date='2020-01-23'):
    config = parse_config.read_config()
    globals().update(config)

    # t_gap = date(2020, 1, 23)
    t1 = date.fromisoformat(start_date)
    t2 = date.fromisoformat(end_date)
    # if(t1 < t_gap):
    #     # file_name = 'epi_initial-67.xlsx'
    #     df = pd.read_excel(os.path.join(DATA_ROOT, file_name), sheet_name='城市确诊人数', index_col=0)
    #     df = df.fillna(0)
    #     city_infect = df.set_index('city').loc[city]
    #     city_I = pd.DataFrame({'infect': city_infect[1:]})
    #     I = city_I.loc[t1:t2]['infect'].values
    # else:
    file_name = '2.csv'
    df = pd.read_csv(os.path.join(DATA_ROOT, file_name), index_col=0)
    df.index = pd.to_datetime(df.index)
    I = df.loc[t1:t2]['infect'].values
    return I

def load_U(city='武汉市', start_date='2020-01-18', end_date='2020-01-23', offset=7):
    config = parse_config.read_config()
    globals().update(config)
    
    t1 = date.fromisoformat(start_date)
    t2 = date.fromisoformat(end_date)
    I_t1 = t1 + timedelta(days=offset)
    I_t2 = t2 + timedelta(days=offset)
    # print(t1, t2, I_t1, I_t2)
    
    file_name_1, file_name_2 = '1.csv', '2.csv'
    df = pd.read_csv(os.path.join(DATA_ROOT, file_name_1), index_col=0)
    df.index = pd.to_datetime(df.index)
    if(model_name == 'SUIR'):
        I_all = df.loc[I_t1:I_t2]['infect'].values
    else:
        I_all = df.loc[t1:t2]['infect'].values

    df = pd.read_csv(os.path.join(DATA_ROOT, file_name_2), index_col=0)
    df.index = pd.to_datetime(df.index)
    I = df.loc[t1:t2]['infect'].values
    # print(len(I))

    U = (I_all - I) #.values
    return U

def load_E(city='武汉市', start_date='2020-01-18', end_date='2020-01-23', offset=7):
    config = parse_config.read_config()
    globals().update(config)
    
    t1 = date.fromisoformat(start_date)
    t2 = date.fromisoformat(end_date)
    # I_t1 = t1 + timedelta(days=offset)
    # I_t2 = t2 + timedelta(days=offset)
    # print(t1, t2, I_t1, I_t2)
    
    file_name_1 = '3.csv' # '1.csv'
    df = pd.read_csv(os.path.join(DATA_ROOT, file_name_1), index_col=0)
    df.index = pd.to_datetime(df.index)
    E = df.loc[t1:t2]['infect'].values
    # E = df.loc[I_t1:I_t2]['infect'].values # t + offset 发病
    # U = load_U(start_date=start_date, end_date=end_date) # t 未确诊
    # U = df.loc[t1:t2]['infect'].values
    return E
