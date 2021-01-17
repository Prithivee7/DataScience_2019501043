import pandas as pd

csv_daily = pd.read_csv('daily.csv')

csv_daily.sort_values(['Gain or Loss'],inplace=True)
#Heatmap function
top_25_daily = csv_daily.head(25)
print(top_25_daily.corr())

bottom_25_points = csvd.tail(25)
print(bottom_25_points.corr())

csv_weekly = pd.read_csv('weekly.csv')

csv_weekly.sort_values(['Gain or Loss'],inplace=True)

top_25_weekly = csv_weekly.head(25)
print(top_25_weekly.corr())

bottom_25_weekly = csv_weekly.tail(25)
print(bottom_25_weekly.corr())

csv_monthly = pd.read_csv('monthly.csv')

csv_monthly.sort_values(['Gain or Loss'],inplace=True)

top_25_monthly = csv_monthly.head(25)
print(top_25_monthly.corr())

bottom_25_monthly = csv_monthly.tail(25)
print(bottom_25_monthly.corr())
