import pandas as pd

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def convert_csv_monthly():
    file_name = 'monthly.csv'
    df = pd.read_csv(file_name)
    ss = df['Close'].values.tolist()
    fin = []
    for i in ss:
        if(isfloat(i) == True):
            fin.append(i)    
    return fin

def cal_close(fin):
    final = [0 for i in range(len(fin))]
    for i in range(1,len(fin)):
        s = final[i] - final[i-1]
        final[i] = (s / final[i]) * 100
    return final



def convert_csv_df_daily():    
    file_name = 'daily.csv'
    df = pd.read_csv(file_name)

def convert_csv_df_weekly():
    file_name = 'weekly.csv'
    df = pd.read_csv(file_name)

# convert_csv_df_daily()
# convert_csv_df_weekly()
convert_csv_monthly()
