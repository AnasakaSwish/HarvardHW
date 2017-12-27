'''
Created on 10 nov. 2017

@author: anase
'''
from fnmatch import fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
from dask.dataframe.core import DataFrame
urlRCP='http://www.realclearpolitics.com/epolls/2010/governor/2010_elections_governor_map.html'

def get_poll_lxml(poll_id):
    url="http://charts.realclearpolitics.com/charts/"+str(poll_id)+".xml"
    return requests.get(url).text
def _strip(s):
    return re.sub(r'[\W_]+','',s)
def plot_colors(xml):
    soup=BeautifulSoup(xml,'lxml')
    result={}
    for graph in soup.find_all("graph"):
        title=_strip(graph['title'])
        result[title]=graph["color"]

    return result
def rcp_poll_data(xml):
    soup=BeautifulSoup(xml,"lxml")
    dictframe={}
    datelist=[]
    
    for graph in soup.graphs.find_all("graph"):
        valuelist=[]

        for value in graph.find_all("value"):
            valuelist.append(value.string)
        dictframe[graph["title"]]=valuelist
    
    for value in soup.series.find_all('value'):
        datelist.append(value.string)
    dictframe['date']=pd.to_datetime(datelist)
    result=pd.DataFrame(dictframe)

    return result

def poll_plot(poll_id):
    """
    Make a plot of an RCP Poll over time
    
    Parameters
    ----------
    poll_id : int
        An RCP poll identifier
    """

    # hey, you wrote two of these functions. Thanks for that!
    xml = get_poll_lxml(poll_id)
    data = rcp_poll_data(xml)
    colors = plot_colors(xml)

    #remove characters like apostrophes
   
    data = data.rename(columns = {c: _strip(c) for c in data.columns})
    
    
        
    for c in list(colors.keys()):
        
        data[c]=data[c].astype('float64')
        norm = data[c].sum() / 100 
        data[c] = data[c] / norm
      
    for label, color in list(colors.items()):
        
        plt.plot(data.date, data[label], color=color, label=label)        
          
    plt.xticks(rotation=70)
    plt.legend(loc='best')
    plt.xlabel("Date")
    plt.ylabel("Normalized Poll Percentage")
    plt.show()

def find_governor_races(urlRCP):
    soup=BeautifulSoup(requests.get(urlRCP).text,"lxml")
    linktables=soup.find("div",{"id":"mymap"})
    hrefs=[]
    for a in linktables.find_all("a"):
        hrefs.append(a["href"])
    hrefs=list(filter(None,hrefs))
    urlizer= lambda x: "https://www.realclearpolitics.com"+x
    hrefs=list(map(urlizer,hrefs))
    return hrefs

def race_result(url):
    results={}
    soup=BeautifulSoup(requests.get(url).text,"lxml")
    table=soup.find("div",{"id":"polling-data-rcp"})
    
    if len(table.find_all("th"))== 6:
        results[table.find_all("th")[3].string[:-4]]=float(table.find_all("td")[3].string)
        results[table.find_all("th")[4].string[:-4]]=float(table.find_all("td")[4].string)
    else:
        if len(table.find_all("th"))== 7:
            results[table.find_all("th")[3].string[:-4]]=float(table.find_all("td")[3].string)
            results[table.find_all("th")[4].string[:-4]]=float(table.find_all("td")[4].string)
            if "(" not in table.find_all("th")[5].string:
                
                results[table.find_all("th")[5].string]=float(table.find_all("td")[5].string.strip())
            else:
                results[table.find_all("th")[5].string[:-4]]=float(table.find_all("td")[5].string.strip())
    
    return results

def id_from_url(url):
    """Given a URL, look up the RCP identifier number"""
    return url.split('-')[-1].split('.html')[0]

def plot_race(url):
    """Make a plot summarizing a senate race
    
    Overplots the actual race results as dashed horizontal lines
    """
    #hey, thanks again for these functions!
    idrace = id_from_url(url)
    xml = get_poll_lxml(idrace)    
    colors = plot_colors(xml)

    if len(colors) == 0:
        return
    
    #really, you shouldn't have
    result = race_result(url)
    
    poll_plot(idrace)
    plt.xlabel("Date")
    plt.ylabel("Polling Percentage")
    for r in result:
        plt.axhline(result[r], color=colors[_strip(r)], alpha=0.6, ls='--')
        
def party_from_color(color):
    if color in ['#0000CC', '#3B5998']:
        return 'democrat'
    if color in ['#FF0000', '#D30015']:
        return 'republican'
    return 'other'


def error_data(url):
    """
    Given a Governor race URL, download the poll data and race result,
    and construct a DataFrame with the following columns:
    
    candidate: Name of the candidate
    forecast_length: Number of days before the election
    percentage: The percent of poll votes a candidate has.
                Normalized to that the canddidate percentages add to 100%
    error: Difference between percentage and actual race reulst
    party: Political party of the candidate
    
    The data are resampled as necessary, to provide one data point per day
    """
    
    idr = id_from_url(url)
    xml = get_poll_lxml(idr)
    
    colors = plot_colors(xml)
    if len(colors) == 0:
        return pd.DataFrame()
    
    df = rcp_poll_data(xml)
    result = race_result(url)
    
    #remove non-letter characters from columns
    df = df.rename(columns={c: _strip(c) for c in df.columns})
    for k, v in list(result.items()):
        result[_strip(k)] = v 
    
    candidates = [c for c in df.columns if c is not 'date']
        
    #turn into a timeseries...
    df.index = df.date
    
    #...so that we can resample at regular, daily intervals
    df = df.resample('D',convention='start').asfreq()
    df = df.dropna()
    
    #compute forecast length in days
    #(assuming that last forecast happens on the day of the election, for simplicity)
    forecast_length = (df.date.max() - df.date).values
    forecast_length = forecast_length / np.timedelta64(1, 'D')  # convert to number of days
    
    #compute forecast error
    errors = {}
    normalized = {}
    poll_lead = {}

    
    for c in candidates:
        #turn raw percentage into percentage of poll votes
        
        corr = (df[c].astype('float64') / df[candidates].astype('float64').sum(axis=1))*100.
        
        err = corr - result[_strip(c)]
        
    
        normalized[c] = corr
        errors[c] = err
        
    n = forecast_length.size
    
    result = {}
    result['percentage'] = np.hstack(normalized[c] for c in candidates)
    result['error'] = np.hstack(errors[c] for c in candidates)
    result['candidate'] = np.hstack(np.repeat(c, n) for c in candidates)
    result['party'] = np.hstack(np.repeat(party_from_color(colors[_strip(c)]), n) for c in candidates)
    result['forecast_length'] = np.hstack(forecast_length for _ in candidates)
    
    result = pd.DataFrame(result)
    
    return result

def all_error_data():
    errorlist=[]
    for race in find_governor_races(urlRCP):
        errorlist.append(error_data(race))
    result=pd.concat(errorlist,ignore_index=True)
    return result

# all_error_data().to_csv("ErrorData.csv")
# errors=pd.read_csv("ErrorData.csv")
#  
# errors.error.hist(bins=50)
# plt.xlabel=("Polling error")
# plt.ylabel=("N")
# plt.show()

#error_data('https://www.realclearpolitics.com/epolls/2013/governor/va/virginia_governor_cuccinelli_vs_mcauliffe-3033.html')
    
    
    
    