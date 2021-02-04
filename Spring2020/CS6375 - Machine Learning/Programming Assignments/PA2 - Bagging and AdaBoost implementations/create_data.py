import pandas as pd
import numpy as np

names = ['classes','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing',
'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',
'stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type',
'spore-print-color','population','habitat']
df = pd.read_csv('agaricus-lepiota.data',delimiter=',',header=None,names=names)
print(df.head())

df = df[['bruises?','cap-shape','cap-surface','cap-color','classes','odor','gill-attachment','gill-spacing',
'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',
'stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type',
'spore-print-color','population','habitat']]

class_variable = df['bruises?']
df = df.drop(columns=['stalk-root'])
df = df.rename(columns={"classes":"edibility"})

print(df.head())

df.replace({'edibility': {'e': 0, 'p': 1}},inplace=True)
df.replace({'bruises?': {'t': 0, 'f': 1}},inplace=True)
df.replace({'cap-shape': {'b': 0, 'c': 1,'x':2,'f':3,'k':4,'s':5}},inplace=True)
df.replace({'cap-surface': {'f': 0, 'g': 1,'y':2,'s':3}},inplace=True)
df.replace({'cap-color': {'n': 0, 'b': 1,'c':2,'g':3,'r':4,'p':5,'u':6,'e':7,'w':8,'y':9}},inplace=True)
df.replace({'odor': {'a': 0, 'l': 1,'c':2,'y':3,'f':4,'m':5,'n':6,'p':7,'s':8}},inplace=True)
df.replace({'gill-attachment': {'a': 0, 'd': 1,'f':2,'n':3}},inplace=True)
df.replace({'gill-spacing': {'c': 0, 'w': 1,'d':2}},inplace=True)
df.replace({'gill-size': {'b': 0, 'n': 1}},inplace=True)
df.replace({'gill-color': {'k': 0, 'n': 1,'b':2,'h':3,'g':4,'r':5,'o':6,'p':7,'u':8,'e':9,'w':10,'y':11}},inplace=True)
df.replace({'stalk-shape': {'e': 0, 't': 1}},inplace=True)
df.replace({'stalk-surface-above-ring': {'f': 0, 'y': 1,'k':2,'s':3}},inplace=True)
df.replace({'stalk-surface-below-ring': {'f': 0, 'y': 1,'k':2,'s':3}},inplace=True)
df.replace({'stalk-color-above-ring': {'n': 0, 'b': 1,'c':2,'g':3,'o':4,'p':5,'e':6,'w':7,'y':8}},inplace=True)
df.replace({'stalk-color-below-ring': {'n': 0, 'b': 1,'c':2,'g':3,'o':4,'p':5,'e':6,'w':7,'y':8}},inplace=True)
df.replace({'veil-type': {'p': 0, 'u': 1}},inplace=True)
df.replace({'veil-color': {'n': 0, 'o': 1,'w':2,'y':3}},inplace=True)
df.replace({'ring-number': {'n': 0, 'o': 1,'t':2}},inplace=True)
df.replace({'ring-type': {'c': 0, 'e': 1,'f':2,'l':3,'n':4,'p':5,'s':6,'z':7}},inplace=True)
df.replace({'spore-print-color': {'k': 0, 'n': 1,'b':2,'h':3,'r':4,'o':5,'u':6,'w':7,'y':8}},inplace=True)
df.replace({'population': {'a': 0, 'c': 1,'n':2,'s':3,'v':4,'y':5}},inplace=True)
df.replace({'habitat': {'g': 0, 'l': 1,'m':2,'p':3,'u':4,'w':5,'d':6}},inplace=True)


print(df.head())
df.to_csv('filter.csv',index=False,header=False)
np.random.seed(7)
msk = np.random.rand(len(df)) < 0.75
train = df[msk]
test = df[~msk]

print(len(train))
print(len(test))

train.to_csv('train',index=False,header=False)
test.to_csv('test',index=False,header=False)