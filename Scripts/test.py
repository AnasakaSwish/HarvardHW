'''
Created on 14 déc. 2017

@author: anase
'''
import pandas as pd
import json
from bs4 import BeautifulSoup
import requests
import re
from astropy.units import day


df=pd.read_csv("Alldata.csv",sep=";",encoding="utf-8-sig").set_index(["Films","Critic"])

y=[1 if x>2 else 0 for x in df.Note.values]
print(mean(y))

# # with open("Allocinecritics.json","r",encoding='utf-8') as json_data:
# #     data= json.load(json_data)
# # print(data)
# 
# url="http://www.allocine.fr/film/fichefilm_gen_cfilm=4936.html"
#  
# soup=BeautifulSoup(requests.get(url).text,"lxml")
# print(soup.find("span",{"class":re.compile('.*blue-link.*')}).string)
# 
# date="12 juillet 2006"
# 
# def monthtonumber(month):
#     return {"janvier":"01","février":"02","mars":"03","avril":"04","mai":"05","juin":"06","juillet":"07","août":"08","septembre":"09","octobre":"10","novembre":"11","décembre":"12"}[month]
# 
# 
# def traductiondate(date):
#     day=date.split(" ")[0]
#     if len(day)<2:
#         day=str(0)+day
#     year=date.split(" ")[2]
#     month=monthtonumber(date.split(" ")[1])
#     dateiso=year+"-"+month+"-"+day
#     
#     return dateiso
# 
# print(traductiondate(date))
# # print(type(float(soup.find("div",{"class":re.compile('.*rating-mdl.*')})["class"][1][1:-1])))
# 
# # s="/film/agenda/sem-2017-12-06/"
# # print(s.split("/")[3][4:])
