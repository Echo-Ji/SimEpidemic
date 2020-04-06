import datetime as dt

def str_to_dt(datestr):
    return dt.datetime.strptime(datestr, '%Y-%m-%d').date()
    
def dt_to_str(datedt):
    return datedt.strftime('%Y-%m-%d')