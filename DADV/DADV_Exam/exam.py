from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_data(url_gics_sectors):
    html_content = requests.get(url_gics_sectors).text
    soup = BeautifulSoup(html_content, "lxml") 
    find_table = soup.find('table', class_='wikitable sortable')
    j = 0
    full_val = []
    for link in find_table.find_all('tr'):
        j += 1
        if(j == 1):
            continue
        i = 0
        d = {}
        for ele in link.find_all('td'):
            i += 1
            if(i == 1):
                st = ele.text
                st = st[:-1]
                d['symbol'] = st
            # elif(i == 2):
            #     d['security'] = ele.text
            # elif(i == 3):
            #     d['sec_filings'] = ele.text
            elif(i == 4):
                d['gics_sector'] = ele.text
            # elif(i == 5):
            #     d['gics_sub_industry'] = ele.text
            # elif(i == 6):
            #     d['hq'] = ele.text
            # elif(i == 7):
            #     d['date'] = ele.text
            # elif(i == 8):
            #     d['cik'] = ele.text
            # elif(i == 9):
            #     d['founded'] = ele.text
        
        full_val.append(d)
    df = pd.DataFrame(full_val)
    # print(df)
    df.to_csv('file4.csv')
    return full_val


def build_data_bar_graph(full_val):
    dp_top = {}
    dp_bottom = {}
    top = [full_val[i] for i in range(len(full_val)//2)]
    bottom = [full_val[i] for i in range(len(full_val)//2,len(full_val))]
    for q in top:
        string = q['gics_sector']
        if(string in dp_top):
            dp_top[string] += 1
        else:
            dp_top[string] = 1
    
    for q in bottom:
        string = q['gics_sector']
        
        if(string in dp_bottom):
            dp_bottom[string] += 1
        else:
            dp_bottom[string] = 1
    
    return dp_top,dp_bottom


def bar_graph(dp_top,dp_bottom):
    dp_top = dict(sorted(dp_top.items(), key=lambda item: item[1],reverse=True)[:10])
    dp_bottom = dict(sorted(dp_bottom.items(), key=lambda item: item[1],reverse=True)[:10])

    dp_new = []
    top_new = []
    bottom_new = []
    for i in dp_top:
        if(i in dp_bottom):
            top_new.append(dp_top[i])
            bottom_new.append(dp_bottom[i])
            dp_new.append(i)

    data = []
    data.append(top_new)
    data.append(bottom_new)



    barWidth = 0.25
    fig = plt.subplots(figsize=(25,9))
    br1 = np.arange(len(top_new))
    br2 = [x + barWidth for x in br1]

    plt.bar(br1, top_new, color ='g', width = barWidth, 
        edgecolor ='grey', label ='Top 25 companies') 
    plt.bar(br2, bottom_new, color ='r', width = barWidth, 
        edgecolor ='grey', label ='Bottom 25 companies') 
    plt.legend(labels=['Top 25', 'Bottom 25'])
    plt.xlabel('Sectors', fontweight ='bold') 
    plt.ylabel('Frequency', fontweight ='bold')
    plt.xticks([r + barWidth for r in range(len(top_new))], 
    dp_new) 

    plt.show()

    


def question_4(url_gics_sectors):
    full_val = extract_data(url_gics_sectors)
    dp_top, dp_bottom = build_data_bar_graph(full_val)
    bar_graph(dp_top,dp_bottom)



url_gics_sectors = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
question_4(url_gics_sectors)