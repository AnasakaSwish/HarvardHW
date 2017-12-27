'''
Created on 1 déc. 2017

@author: anase
'''
import json
import requests
import time
import re
import numpy as np


from bs4 import BeautifulSoup

films={}

def monthtonumber(month):
    return {"janvier":"01","février":"02","mars":"03","avril":"04","mai":"05","juin":"06","juillet":"07","août":"08","septembre":"09","octobre":"10","novembre":"11","décembre":"12"}[month]

def traductiondate(date):
    
    day=date.split(" ")[0]
    if len(day)<2:
        day=str(0)+day
    year=date.split(" ")[2]
    month=monthtonumber(date.split(" ")[1])
    dateiso=year+"-"+month+"-"+day

    return dateiso

for i in range(1,141):
    page=str(i)
    url = "http://www.allocine.fr/films/presse/genre-13005/?page="+page
    soup=BeautifulSoup(requests.get(url).text,"lxml")  
     
    for film in soup.find_all("li",{"class":"hred"}):
        title=film.find("h2",{"class":"meta-title"}).a.string
        print("Scrapping "+title)
        href=film.find("h2",{"class":"meta-title"}).a["href"]
        noteGL=film.find("span",{"class":"stareval-note"}).string.strip()
        idfilm=href.split("=")[1].split(".")[0]
        urld="http://www.allocine.fr"+href
        soup3=BeautifulSoup(requests.get(urld).text,"lxml")
        date=soup3.find("span",{"class":re.compile('.*blue-link.*')}).string
        
        urlc="http://www.allocine.fr/film/fichefilm-"+str(idfilm)+"/critiques/presse/"
        soup2=BeautifulSoup(requests.get(urlc).text,"lxml")
        critiques={}
        for critique in soup2.find_all("div",{"class":"item hred"}):
             
            critic=critique.h2.span.string
            notec=float(critique.find("div",{"class":re.compile('.*rating-mdl.*')})["class"][1][1:-1])
            desc=critique.p.string.strip().replace(";"," ")
            critiques[critic]={"Note":notec,"Critique":desc}
        try:
            films[title]={"Note Globale":float(noteGL.replace(",",".")),"Critiques":critiques,"Date":traductiondate(date)}
            
        except IndexError:
            films[title]={"Note Globale":float(noteGL.replace(",",".")),"Critiques":critiques,"Date":np.nan}
            
        time.sleep(1)
    print("Page "+str(i)+" scrapped !")   
print("All done. Writing to json")
with open("Allocinecritics1.json","w",encoding='utf-8') as f:
            json.dump(films,f,ensure_ascii=False)


# with open("Allocinecritics.json","r",encoding='utf-8') as json_data:
#     d = json.load(json_data)
#     pprint(d)