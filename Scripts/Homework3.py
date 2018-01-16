'''
Created on 8 déc. 2017

@author: anase
'''

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cycler
#
# pd.set_option('display.width', 500)
# pd.set_option('display.max_columns', 30)
#
# # set some nicer defaults for matplotlib
# from matplotlib import rcParams
#
# # these colors come from colorbrewer2.org. Each is an RGB triplet
# dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
#                 (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
#                 (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
#                 (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
#                 (0.4, 0.6509803921568628, 0.11764705882352941),
#                 (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
#                 (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
#                 (0.4, 0.4, 0.4)]
#
# rcParams['figure.figsize'] = (10, 6)
# rcParams['figure.dpi'] = 150
# rcParams['axes.prop_cycle'] = cycler("color", dark2_colors)
# rcParams['lines.linewidth'] = 2
# rcParams['axes.grid'] = False
# rcParams['axes.facecolor'] = 'white'
# rcParams['font.size'] = 14
# rcParams['patch.edgecolor'] = 'none'
#
#
# def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
#     """
#     Minimize chartjunk by stripping out unnecessary plot borders and axis ticks
#
#     The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
#     """
#     ax = axes or plt.gca()
#     ax.spines['top'].set_visible(top)
#     ax.spines['right'].set_visible(right)
#     ax.spines['left'].set_visible(left)
#     ax.spines['bottom'].set_visible(bottom)
#
#     # turn off all ticks
#     ax.yaxis.set_ticks_position('none')
#     ax.xaxis.set_ticks_position('none')
#
#     # now re-enable visibles
#     if top:
#         ax.xaxis.tick_top()
#     if bottom:
#         ax.xaxis.tick_bottom()
#     if left:
#         ax.yaxis.tick_left()
#     if right:
#         ax.yaxis.tick_right()


# def get_reviews(title):
#     return pd.DataFrame.from_dict(data.loc[title,"Critiques"],orient="index")
# 
# def frame_reviews(data):
#     frames=[]
#     filmlist=[]
#     df=data.Critiques
#     print(df)

# with open("Allocinecritics1.json","r",encoding='utf-8') as json_data:
#     data= json.load(json_data)

# filmlist=[]
# notedict={}
# datedict={}
# reviewframes=[]
# filmdict={}
# datelist=[]
# i=0
# for film in data.keys():
#     filmlist.append(film)
#     criticlist=[]
#     critdict=data[film]["Critiques"]
#     notedict[film]=data[film]["Note Globale"]
#     datedict[film]=data[film]["Date"]
#     datelist.append(datedict[film])
#     reviewdict={}
#     for critic,review in critdict.items():
#         criticlist.append(critic)
#         reviewdict[critic]=[review["Note"],review["Critique"]]
#        
#     reviewframe=pd.DataFrame.from_dict(reviewdict,orient="index")
#     reviewframe.columns=["Note","Critique"]
#     reviewframe.index.name="Critic"
#     reviewframes.append(reviewframe)
#       
#      
# df1=pd.concat(reviewframes,keys=filmlist)
# df1.index.names=["Films","Critic"]
# df1=df1.sort_index(axis=0,level=0)
# df1.to_csv("concatene.csv",sep=";",mode="w",encoding="utf-8-sig")
#  
# df3=pd.Series(datedict,index=filmlist,name="Date")
# df3=pd.to_datetime(df3,yearfirst=True)
# 
# df5=pd.Series(notedict,index=filmlist,name="Note Globale")
# df4=pd.concat([df5,df3],axis=1,keys=filmlist)
# df4.columns=["Note Globale","Date"]
# df4.index.name="Films"
# 
# df4.to_csv("DateNote.csv",sep=";",mode="w",encoding="utf-8-sig")
# df2.to_csv("Noteglobale.csv",sep=";",mode="w",encoding="utf-8-sig")

# counts=df1.groupby("Critics").count()
# print(counts.sort_values(by=0)[-1:-6:-1])

# df1=pd.read_csv("concatene.csv",sep=";",encoding="utf-8-sig").set_index("Films")
# df2=pd.read_csv("DateNote.csv",sep=";",encoding="utf-8-sig").set_index("Films")
# 
# 
# 
# 
# df1=df1.reset_index()
# #df1.set_index(["Films","Critic"],inplace=True)
# df3=df1.join(df2,on="Films")
# df3.set_index(["Films","Critic"],inplace=True)
# print(df3)


# text = ['Hop on pop', 'Hop off pop', 'Hop Hop hop']
# print "Original text is\n", '\n'.join(text)
# 
# vectorizer = CountVectorizer(min_df=0)
# 
# # call `fit` to build the vocabulary
# vectorizer.fit(text)
# 
# # call `transform` to convert text to a bag of words
# x = vectorizer.transform(text)
# 
# # CountVectorizer uses a sparse array to save memory, but it's easier in this assignment to 
# # convert back to a "normal" numpy array
# x = x.toarray()
# 
# print
# print "Transformed text vector is \n", x
# 
# # `get_feature_names` tracks which word is associated with each column of the transformed x
# print
# print "Words for each feature:"
# print vectorizer.get_feature_names()
# 
# # Notice that the bag of words treatment doesn't preserve information about the *order* of words, 
# # just their frequency

df = pd.read_csv("Alldata.csv", sep=";", encoding="utf-8-sig").set_index(["Films", "Critic"])
df=df[:15000]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def make_xy(critics, vectorizer=None):
    X = critics.Critique.values
    y = [1 if x > 3 else 0 for x in critics.Note.values]

    if vectorizer is None:
        vectorizer = CountVectorizer()

    x = vectorizer.fit_transform(X)
    x = x.tocsc()
    return x, y

vectorizer=CountVectorizer(min_df=0.000010)
X, Y = make_xy(df,vectorizer)
X_train, X_test, y_train, y_test = train_test_split(X, Y)
clf = MultinomialNB(alpha=1.)
clf.fit(X_train, y_train)

# def calibration_plot(clf, X, Y):
#     prob = clf.predict_proba(X)[:, 1]
#     outcome = Y
#     data = pd.DataFrame(dict(prob=prob, outcome=outcome))
#     # group outcomes into bins of similar probability
#     bins = np.linspace(0, 1, 20)
#     cuts = pd.cut(prob, bins)
#     binwidth = bins[1] - bins[0]
#     cal = data.groupby(cuts).outcome.agg(['mean', 'count'])
#     cal['pmid'] = (bins[:-1] + bins[1:]) / 2
#     cal['sig'] = np.sqrt(cal.pmid * (1 - cal.pmid) / cal['count'])
#     ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
#     p = plt.errorbar(cal.pmid, cal['mean'], cal['sig'])
#     plt.plot(cal.pmid, cal.pmid, linestyle='--', lw=1, color='k')
#     plt.ylabel("Empirical P(Fresh)")
#     remove_border(ax)
#
#     # the distribution of P(fresh)
#     ax = plt.subplot2grid((3, 1), (2, 0), sharex=ax)
#
#     plt.bar(left=cal.pmid - binwidth / 2, height=cal['count'],
#             width=.95 * (bins[1] - bins[0]),
#             fc=p[0].get_color())
#
#     plt.xlabel("Predicted P(Fresh)")
#     remove_border()
#     plt.ylabel("Number")
#     plt.legend()


words = np.array(vectorizer.get_feature_names())
#x= np.eye(X_test.shape[1])



# prob = clf.predict_log_proba(X)[:,0]
#
# predict = clf.predict(x)
# ind = np.argsort(probs)
#
# good_words = words[ind[:10]]
# bad_words = words[ind[-10:]]
#
# good_prob = probs[ind[:10]]
# bad_prob = probs[ind[-10:]]
#
# print(
# "Good words\t     P(note>3 | mot)")
# for w, p in zip(good_words, good_prob):
#     print(
#     "%20s" % w, "%0.2f" % (1 - np.exp(p)))
#
# print(
# "Bad words\t     P(note>3 | mot)")
# for w, p in zip(bad_words, bad_prob):
#     print(
#     "%20s" % w, "%0.2f" % (1 - np.exp(p)))

x, y = make_xy(df, vectorizer)

prob = clf.predict_proba(x)[:, 0]
predict = clf.predict(x)

bad_rotten = np.argsort(prob[predict == 0])[:5]
bad_fresh = np.argsort(prob[predict == 1])[-5:]
print(bad_rotten)
print(bad_fresh)
df=df.reset_index()


print ("Mis-predicted Rotten quotes")
print ('---------------------------')

for row in bad_rotten:
    print (df[predict == 0].loc[row,"Critique"])

print ("Mis-predicted fresh quotes")
print ('---------------------------')
for row in bad_fresh:
    print (df[predict == 1].loc[row,"Critique"])
    print (" ")