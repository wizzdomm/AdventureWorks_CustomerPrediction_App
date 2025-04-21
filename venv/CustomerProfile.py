# from msilib.schema import RadioButton
from tkinter import *
import ttkbootstrap as ttk
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd

from ttkbootstrap.constants import *

root = ttk.Window(themename='darkly')
root.title('BikebuyerPrediction')
root.geometry('650x850')
root.resizable(0, 0)
w = 650
h = 850

root.overrideredirect()
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
x = (ws - w) / 2
y = (hs - h) / 4
root.geometry("%dx%d+%d+%d" % (w, h, x, y))


#  functions
# sEnglishEducation = []
# sEnglishOccupation = []
# sRegion = []
# sCommuteDistance = []

def Prediction():
    df = pd.read_csv('vtargetmailmodified.csv')

    Age = float(Age_Entry.get())
    Income = float(YearlyIncome_Entry.get())
    NumberCarsOwned = float(NumberCarsOwned_ComboBox.get())
    TotalChildren = float(TotalChildren_ComboBox.get())

    EnglishEducation = []
    if EnglishEducation_ComboBox.get() == 'PartialHighSchool':
        EnglishEducation = 1.0
    elif EnglishEducation_ComboBox.get() == 'HighSchool':
        EnglishEducation = 2.0
    elif EnglishEducation_ComboBox.get() == 'PartialCollege':
        EnglishEducation = 3.0
    elif EnglishEducation_ComboBox.get() == 'Bachelors':
        EnglishEducation = 4.0
    elif EnglishEducation_ComboBox.get() == 'GraduateDegree':
        EnglishEducation = 5.0

    # EnglishEducation[3] = EnglishEducation_ComboBox.get().replace('Partial High School','1')
    # EnglishEducation[2] = EnglishEducation_ComboBox.get().replace('High School', '2')
    # EnglishEducation[0] = EnglishEducation_ComboBox.get().replace('Bachelors', '4')
    # EnglishEducation[4] = EnglishEducation_ComboBox.get().replace('Partial College', '3')
    # EnglishEducation[1] = EnglishEducation_ComboBox.get().replace('Graduate Degree', '5')

    # EnglishEducation = EnglishEducation_ComboBox.get().replace('Partial High School','1'),EnglishEducation_ComboBox.get().replace('High School', '2'),EnglishEducation_ComboBox.get().replace('Bachelors', '4'),EnglishEducation_ComboBox.get().replace('Partial College', '3'),EnglishEducation_ComboBox.get().replace('Graduate Degree','5')

    # EnglishOccupation = EnglishOccupation_ComboBox.get().replace('Clerical','1'), EnglishOccupation_ComboBox.get().replace('Management','2'), EnglishOccupation_ComboBox.get().replace('Manual','3'),EnglishOccupation_ComboBox.get().replace('Professional','4'),EnglishOccupation_ComboBox.get().replace('Skilled Manual','5')

    EnglishOccupation = 0
    if EnglishOccupation_ComboBox.get() == 'Clerical':
        EnglishOccupation = 1.0
    elif EnglishOccupation_ComboBox.get() == 'Management':
        EnglishOccupation = 2.0
    elif EnglishOccupation_ComboBox.get() == 'Manual':
        EnglishOccupation = 3.0
    elif EnglishOccupation_ComboBox.get() == 'Professional':
        EnglishOccupation = 4.0
    elif EnglishOccupation_ComboBox.get() == 'SkilledManual':
        EnglishOccupation = 5.0
    # EnglishOccupation[0] = EnglishOccupation_ComboBox.get().replace('Clerical', '1')
    # EnglishOccupation[1] = EnglishOccupation_ComboBox.get().replace('Management', '2')
    # EnglishOccupation[2] = EnglishOccupation_ComboBox.get().replace('Manual', '3')
    # EnglishOccupation[3] = EnglishOccupation_ComboBox.get().replace('Professional', '4')
    # EnglishOccupation[4] = EnglishOccupation_ComboBox.get().replace('Skilled Manual', '5')

    # CommuteDistance =CommuteDistance_ComboBox.get().replace('0-1 Miles','1'), CommuteDistance_ComboBox.get().replace('1-2 Miles','2'), CommuteDistance_ComboBox.get().replace('2-5 Miles','3'), CommuteDistance_ComboBox.get().replace('5-10 Miles','4'), CommuteDistance_ComboBox.get().replace('10+ Miles','5')

    CommuteDistance = 0
    if CommuteDistance_ComboBox.get() == '0-1Miles':
        CommuteDistance = 1.0
    elif CommuteDistance_ComboBox.get() == '1-2Miles':
        CommuteDistance = 2.0
    elif CommuteDistance_ComboBox.get() == '2-5Miles':
        CommuteDistance = 3.0
    elif CommuteDistance_ComboBox.get() == '5-10Miles':
        CommuteDistance = 4.0
    elif CommuteDistance_ComboBox.get() == '10+Miles':
        CommuteDistance = 5.0
    # CommuteDistance[0] = CommuteDistance_ComboBox.get().replace('0-1 Miles', '1')
    # CommuteDistance[1] = CommuteDistance_ComboBox.get().replace('1-2 Miles', '2')
    # CommuteDistance[2] = CommuteDistance_ComboBox.get().replace('2-5 Miles', '3')
    # CommuteDistance[3] = CommuteDistance_ComboBox.get().replace('5-10 Miles', '4')
    # CommuteDistance[4] = CommuteDistance_ComboBox.get().replace('10+ Miles', '5')

    # Region = Region_ComboBox.get().replace('Pacific','1'), Region_ComboBox.get().replace('North America','2'), Region_ComboBox.get().replace('Europe','3')
    Region = 0
    if Region_ComboBox.get() == 'Pacific':
        Region = 1.0
    elif Region_ComboBox.get() == 'NorthAmerica':
        Region = 2.0
    elif Region_ComboBox.get() == 'Europe':
        Region = 3.0

    # Region[0] = Region_ComboBox.get().replace('Pacific', '1')
    # Region[1] = Region_ComboBox.get().replace('North America', '2')
    # Region[2] = Region_ComboBox.get().replace('Europe', '3')

    # marital = mMaritalStatus.get().replace('Married','1'),mMaritalStatus.get().replace('Single','0')
    marital = 0
    if mMaritalStatus.get() == 'Married':
        marital = 1.0
    elif mMaritalStatus.get() == 'Single':
        marital = 2.0

    # gender = mGender.get().replace('Female','2'),mGender.get().replace('Male','1')
    gender = 0
    if mGender.get() == 'Female':
        gender = 2.0
    elif mGender.get() == 'Male':
        gender = 1.0

    methods = mMethods.get()

    if methods == 'KNN':
        X = df.iloc[1:, 1:11].astype(float)
        y = df.iloc[1:, 11]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        Classifier = KNeighborsClassifier(n_neighbors=3)
        Classifier.fit(X_train.values, y_train.values)
        KNNPrediction = Classifier.predict([[marital, gender, Income, TotalChildren, EnglishEducation,
                                             EnglishOccupation, NumberCarsOwned, CommuteDistance, Region, Age]])
        Score = f'{Classifier.score(X_test, y_test):.2%}'
        Prediction_label.config(text=KNNPrediction)
        score_label.config(text=Score)
    elif methods == 'NB':
        X = df.iloc[1:, 1:11].astype(float)
        y = df.iloc[1:, 11]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        Classifier = GaussianNB()
        Classifier.fit(X_train.values, y_train.values)
        GaussianNB()
        NBPrediction = Classifier.predict([[marital, gender, Income, TotalChildren, EnglishEducation, EnglishOccupation,
                                            NumberCarsOwned, CommuteDistance, Region, Age]])
        Score = f'{Classifier.score(X_test, y_test):.2%}'
        Prediction_label.config(text=NBPrediction)
        score_label.config(text=Score)
    elif methods == 'DecisionTree':

        # le = LabelEncoder()
        # df['MaritalStatus', 'Gender', 'EnglishEducation', 'EnglishOccupation', 'CommuteDistance', 'Region'] = le.fit_transform(
        #     df['MaritalStatus', 'Gender', 'EnglishEducation', 'EnglishOccupation', 'CommuteDistance', 'Region'])
        X = df[['MaritalStatus', 'Gender', 'YearlyIncome',
                'TotalChildren', 'EnglishEducation',
                'EnglishOccupation', 'NumberCarsOwned',
                'CommuteDistance', 'Region', 'Age']]
        y = df[['BikeBuyer']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        dt = DecisionTreeClassifier(max_depth=3)
        dt_model = dt.fit(X_train, y_train)
        plt.figure(figsize=(14, 12))
        tree.plot_tree(dt_model,
                       feature_names=['MaritalStatus', 'Gender', 'YearlyIncome', 'TotalChildren', 'EnglishEducation',
                                      'EnglishOccupation', 'NumberCarsOwned', 'CommuteDistance', 'Region', 'Age'],
                       class_names=['BikeBuyer', 'NoBikeBuyer'])

        plt.show()


def reset():
    Age_Entry.delete([0], [2])
    YearlyIncome_Entry.delete([0], [5])
    EnglishEducation_ComboBox.current(0)
    EnglishOccupation_ComboBox.current(0)
    NumberCarsOwned_ComboBox.current(0)
    CommuteDistance_ComboBox.current(0)
    Region_ComboBox.current(0)
    TotalChildren_ComboBox.current(0)
    Prediction_label.config(text="")
    score_label.config(text="")


# Labels

Age_label = ttk.Label(text='Age :', font=('Times', 10), bootstyle='default')
Age_label.grid(row=0, column=0, padx=20, pady=20)

YearlyIncome_label = ttk.Label(text='YearlyIncome :', font=('Times', 10), bootstyle='default')
YearlyIncome_label.grid(row=1, column=0, padx=20, pady=10)

# EnglishEducation= ttk.Label(text='Partial High School(1),High School(2),Bachelors(4),Partial College(3),Graduate Degree(5)',font=('Times',10),bootstyle='default')
# EnglishEducation.grid(row=2,column=0,padx=20,pady=20)
EnglishEducation_label = ttk.Label(text='EnglishEducation :', font=('Times', 10), bootstyle='default')
EnglishEducation_label.grid(row=2, column=0, padx=20, pady=20)

# EnglishOccupation=ttk.Label(text='Clerical(1),Management(2),Manual(3),Professional(4),Skilled Manual(5)',font=('Times',10),bootstyle='default')
# EnglishOccupation.grid(row=4,column=0,padx=20,pady=20)
EnglishOccupation_label = ttk.Label(text='EnglishOccupation :', font=('Times', 10), bootstyle='default')
EnglishOccupation_label.grid(row=3, column=0, padx=20, pady=20)

NumberCarsOwned_label = ttk.Label(text='NumberCarsOwned :', font=('Times', 10), bootstyle='default')
NumberCarsOwned_label.grid(row=4, column=0, padx=20, pady=20)

# CommuteDistance=ttk.Label(text='0-1 Miles(1),1-2 Miles(2),2-5 Miles(3),5-10 Miles(4),10+ Miles(5)',font=('Times',10),bootstyle='default')
# CommuteDistance.grid(row=7,column=0,padx=20,pady=20)
CommuteDistance_label = ttk.Label(text='CommuteDistance :', font=('Times', 10), bootstyle='default')
CommuteDistance_label.grid(row=5, column=0, padx=20, pady=20)

# Region=ttk.Label(text='Pacific(1),North America(2),Europe(3)',font=('Times',10),bootstyle='default')
# Region.grid(row=9,column=0,padx=20,pady=20)
Region_label = ttk.Label(text='Region :', font=('Times', 10), bootstyle='default')
Region_label.grid(row=6, column=0, padx=20, pady=20)

TotalChildren_label = ttk.Label(text='TotalChildren :', font=('Times', 10), bootstyle='default')
TotalChildren_label.grid(row=7, column=0, padx=20, pady=20)

Gender_label = ttk.Label(text='Gender :', font=('Times', 10), bootstyle='default')
Gender_label.grid(row=8, column=0, padx=20, pady=20)

MaritalStatus_label = ttk.Label(text='MaritalStatus :', font=('Times', 10), bootstyle='default')
MaritalStatus_label.grid(row=9, column=0, padx=20, pady=20)

Method_label = ttk.Label(text='Method :', font=('Times', 10), bootstyle='default')
Method_label.grid(row=10, column=0, padx=20, pady=20)

prediction_label = ttk.Label(text='prediction:', font=('Times', 10), bootstyle='default')
prediction_label.grid(row=11, column=0, padx=20, pady=20)

Prediction_label = ttk.Label(text='', font=('Times', 10), bootstyle='default')
Prediction_label.grid(row=11, column=1, padx=20, pady=20)

Score_label = ttk.Label(text='Accuracy:', font=('Times', 10), bootstyle='default')
Score_label.grid(row=12, column=0, padx=20, pady=20)

score_label = ttk.Label(text='', font=('Times', 10), bootstyle='default')
score_label.grid(row=12, column=1, padx=20, pady=20)

# End Labels


# Entry

Age_Entry = ttk.Entry(font=('Times', 10), bootstyle='success')
Age_Entry.grid(row=0, column=1, padx=20, pady=20)
YearlyIncome_Entry = ttk.Entry(font=('Times', 10), bootstyle='success')
YearlyIncome_Entry.grid(row=1, column=1, padx=20, pady=10)

# RadioButton

Gender = ['Male', 'Female']
MaritalStatus = ['Married', 'Single']
Methods = ['NB', 'KNN', 'DecisionTree']
mGender = StringVar()
mMaritalStatus = StringVar()
mMethods = StringVar()

for i in Gender:
    if i == 'Male':
        gender = ttk.Radiobutton(root, bootstyle='danger', variable=mGender, text=i, value=i).grid(row=8, column=1,
                                                                                                   padx=20, pady=10,
                                                                                                   sticky='w')
    else:
        gender = ttk.Radiobutton(root, bootstyle='danger', variable=mGender, text=i, value=i).grid(row=8, column=1,
                                                                                                   padx=20, pady=10,
                                                                                                   sticky='e')

for g in MaritalStatus:
    if g == 'Married':
        ttk.Radiobutton(root, bootstyle='danger', variable=mMaritalStatus, text=g, value=g).grid(row=9, column=1,
                                                                                                 padx=20, pady=10,
                                                                                                 sticky='w')
    else:
        ttk.Radiobutton(root, bootstyle='danger', variable=mMaritalStatus, text=g, value=g).grid(row=9, column=1,
                                                                                                 padx=20, pady=10,
                                                                                                 sticky='e')

for J in Methods:
    if J == 'NB':
        ttk.Radiobutton(root, bootstyle='danger', variable=mMethods, text=J, value=J).grid(row=10, column=1, padx=20,
                                                                                           pady=10, sticky='w')
    elif J == 'KNN':
        ttk.Radiobutton(root, bootstyle='danger', variable=mMethods, text=J, value=J).grid(row=10, column=1,
                                                                                           padx=20, pady=10, sticky='e')
    elif J == 'DecisionTree':
        ttk.Radiobutton(root, bootstyle='danger', variable=mMethods, text=J, value=J).grid(row=10, column=2,
                                                                                           padx=20, pady=10, sticky='w')

# Combo box Values

# EnglishEducation=[1,2,3,4,5]
# EnglishOccupation=[1,2,3,4,5]
# NumberCarsOwned=[0,1,2,3,4]
# CommuteDistance=[1,2,3,4,5]
# Region=[1,2,3]
# TotalChildren=[0,1,2,3,4,5]


EnglishEducation = ['Bachelors', 'GraduateDegree', 'HighSchool', 'PartialHighSchool', 'PartialCollege']
EnglishOccupation = ['Clerical', 'Management', 'Manual', 'Professional', 'SkilledManual']
Region = ['Pacific', 'NorthAmerica', 'Europe']
CommuteDistance = ['0-1Miles', '1-2Miles', '2-5Miles', '5-10Miles', '10+Miles']
NumberCarsOwned = [0, 1, 2, 3, 4]
TotalChildren = [0, 1, 2, 3, 4, 5]

# EnglishEducation=['Bachelors':4,'GraduateDegree':5,'HighSchool':2,'PartialHighSchool':1,'PartialCollege':3}
# EnglishOccupation={'Clerical':1,'Management':2,'Manual':3,'Professional':4,'SkilledManual':5}
# NumberCarsOwned=['0','1','2','3','4']
# CommuteDistance={'0-1Miles':1,'1-2Miles':2,'C':3,'5-10Miles':4,'10+Miles':5}
# Region={'Pacific':1,'NorthAmerica':2,'Europe':3}
# TotalChildren=['0','1','2','3','4','5']


# Combo box

EnglishEducation_ComboBox = ttk.Combobox(root, bootstyle='success', values=EnglishEducation)
EnglishEducation_ComboBox.grid(row=2, column=1, padx=20, pady=10)
EnglishEducation_ComboBox.current(0)

EnglishOccupation_ComboBox = ttk.Combobox(root, bootstyle='success', values=EnglishOccupation)
EnglishOccupation_ComboBox.grid(row=3, column=1, padx=20, pady=10)
EnglishOccupation_ComboBox.current(0)

NumberCarsOwned_ComboBox = ttk.Combobox(root, bootstyle='success', values=NumberCarsOwned)
NumberCarsOwned_ComboBox.grid(row=4, column=1, padx=20, pady=10)
NumberCarsOwned_ComboBox.current(0)

CommuteDistance_ComboBox = ttk.Combobox(root, bootstyle='success', values=CommuteDistance)
CommuteDistance_ComboBox.grid(row=5, column=1, padx=20, pady=10)
CommuteDistance_ComboBox.current(0)

Region_ComboBox = ttk.Combobox(root, bootstyle='success', values=Region)
Region_ComboBox.grid(row=6, column=1, padx=20, pady=10)
Region_ComboBox.current(0)

TotalChildren_ComboBox = ttk.Combobox(root, bootstyle='success', values=TotalChildren)
TotalChildren_ComboBox.grid(row=7, column=1, padx=20, pady=10)
TotalChildren_ComboBox.current(0)

# Buttons

Run_Button = ttk.Button(root, text='RUN', width=20, bootstyle='success,outline', command=Prediction)
Run_Button.grid(row=11, column=2, padx=20, pady=20)

reset_Button = ttk.Button(root, text='Reset', width=20, bootstyle='success,outline', command=reset)
reset_Button.grid(row=12, column=2, padx=20, pady=20)

root.mainloop()
