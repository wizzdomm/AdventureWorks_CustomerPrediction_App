import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score

import tkinter as Tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox as msg
from tkcalendar import DateEntry
#--------------------------------------------------

Form = Tk()
Form.geometry('800x800')
right = int(Form.winfo_screenwidth() / 2 - 600 / 2)
down = int(Form.winfo_screenheight() / 2 - 600 / 2)
Form.geometry('+{}+{}'.format(right, down))
# Form.resizable(0, 0)
Form.title('Form')

def check():

    data = pd.read_csv('datasets/CustomerList.csv')
    Age = float(entAge.get())
    Income = float(entYearlyIncome.get())
    NumberCarsOwned = float(entNumberCarsOwned.get())
    TotalChildren = float(entTotalChildren.get())
    NumberChildrenAtHome = float(entNumberChildrenAtHome.get())
    # Birthdate = entBirthDate.get()
    # DateFirstPurchase = entDateFirstPurchase.get()

    EnglishEducation = []
    if entEnglishEducation.get() == 'PartialHighSchool':
        EnglishEducation = 1.0
    elif entEnglishEducation.get() == 'HighSchool':
        EnglishEducation = 2.0
    elif entEnglishEducation.get() == 'PartialCollege':
        EnglishEducation = 3.0
    elif entEnglishEducation.get() == 'Bachelors':
        EnglishEducation = 4.0
    elif entEnglishEducation.get() == 'GraduateDegree':
        EnglishEducation = 5.0

    EnglishOccupation = 0
    if entEnglishOccupation.get() == 'Clerical':
        EnglishOccupation = 1.0
    elif entEnglishOccupation.get() == 'Management':
        EnglishOccupation = 2.0
    elif entEnglishOccupation.get() == 'Manual':
        EnglishOccupation = 3.0
    elif entEnglishOccupation.get() == 'Professional':
        EnglishOccupation = 4.0
    elif entEnglishOccupation.get() == 'SkilledManual':
        EnglishOccupation = 5.0

    CommuteDistance = 0
    if entCommuteDistance.get() == '0-1Miles':
        CommuteDistance = 1.0
    elif entCommuteDistance.get() == '1-2Miles':
        CommuteDistance = 2.0
    elif entCommuteDistance.get() == '2-5Miles':
        CommuteDistance = 3.0
    elif entCommuteDistance.get() == '5-10Miles':
        CommuteDistance = 4.0
    elif entCommuteDistance.get() == '10+Miles':
        CommuteDistance = 5.0

    Region = 0
    if entRegion.get() == 'Pacific':
        Region = 1.0
    elif entRegion.get() == 'NorthAmerica':
        Region = 2.0
    elif entRegion.get() == 'Europe':
        Region = 3.0

    marital = 0
    if mMaritalStatus.get() == 'Married':
        marital = 1.0
    elif mMaritalStatus.get() == 'Single':
        marital = 2.0

    gender = 0
    if mGender.get() == 'Female':
        gender = 2.0
    elif mGender.get() == 'Male':
        gender = 1.0
    Flag = 0
    if entHouseOwnerFlag == 1:
        Flag == 1.0
    elif entHouseOwnerFlag == 0:
        Flag == 0

    data.drop(['Title','MiddleName','Suffix','AddressLine2','FirstName','LastName',
               'NameStyle','AddressLine1','EmailAddress','CustomerKey','GeographyKey','CustomerAlternateKey','Phone','SpanishEducation','SpanishOccupation',
               'FrenchOccupation','FrenchEducation','BirthDate','DateFirstPurchase'],
              axis =1,inplace = True)
    df_num = data[['YearlyIncome','TotalChildren','NumberChildrenAtHome','HouseOwnerFlag','NumberCarsOwned',
                  'Age','BikeBuyer']]
    df_cat = data.select_dtypes(np.object_)
    # print(data.shape)

    label_encoder = LabelEncoder()
    # print(label_encoder)
    # print(df_cat.columns)
    for i in df_cat:
        df_cat[i] = label_encoder.fit_transform(df_cat[i])

    data_n = pd.concat([df_cat,df_num],axis=1)
    # print(data_n.info())

    X = data_n.drop(['BikeBuyer'],axis = 1)
    y = data_n['BikeBuyer']
    # print(X.shape,y.shape,end='\n\n')

    scaler = StandardScaler()
    # print(scaler)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    rf = RandomForestClassifier(max_depth=3)
    cv = cross_val_score(rf, X_train, y_train, cv=5)
    model = rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)

    # result = rf.predict_proba([[10/6/1971,'M','M','Bachelors','Professional',2,1,0,51,'1/19/2011','1-2 Miles','Pacific','90000',0]])
    # print(result)
    result = rf.predict_proba([[Age,Income,NumberCarsOwned,TotalChildren,NumberChildrenAtHome,EnglishEducation,EnglishOccupation,gender,
                                CommuteDistance,Region,marital,Flag]])
    msg.showinfo('Prediction',str(result))
    # tree.plot_tree(model,feature_names=['Age','Income','NumberCarsOwned','TotalChildren','NumberChildrenAtHome','EnglishEducation','EnglishOccupation','gender',
    #                             'CommuteDistance','Region','marital','Flag'],
    #                class_names=['BikeBuyer'])
    # plt.show()


#MODELING
# log = LogisticRegression()
# cv = cross_val_score(log,X_train,y_train,cv=5)
# print('LogisticRegression: ',cv.mean())         #77.54

# gnb = GaussianNB()
# cv = cross_val_score(gnb,X_train,y_train,cv = 5)
# print('GNB: ',cv.mean())                        #77.54

# dt = tree.DecisionTreeClassifier()
# cv = cross_val_score(dt,X_train,y_train,cv = 5)
# print('DecisionTree: ',cv.mean())               #84.04

# knn = KNeighborsClassifier()
# cv = cross_val_score(knn,X_train,y_train,cv = 5)
# print('KNN: ',cv.mean())                        #80.79

# rf = RandomForestClassifier()
# cv = cross_val_score(rf,X_train,y_train,cv = 5)
# print('RandomForest: ',cv.mean())               #88.64

# lblBirthDate = Label(Form, text='BirthDate:')
# lblBirthDate.grid(row=0, column=0, padx=10, pady=10)
#
# # txtBirthDate = StringVar()
# entBirthDate = DateEntry(Form, width=30)
# entBirthDate.grid(row=0, column=1, padx=10, pady=10)

lblMaritalStatus = Label(Form, text='MaritalStatus:')
lblMaritalStatus.grid(row=1, column=0, padx=10, pady=10)

mMaritalStatus = StringVar()
MaritalStatus = ['Married', 'Single']
for g in MaritalStatus:
    if g == 'Married':
        ttk.Radiobutton(Form, variable=mMaritalStatus, text=g, value=g).grid(row=1, column=1, padx=10, pady=10,
                                                                                                 sticky='w')
    else:
        ttk.Radiobutton(Form, variable=mMaritalStatus, text=g, value=g).grid(row=1, column=1, padx=10, pady=10,
                                                                                                 sticky='e')
# entMaritalStatus = ttk.Radiobutton(Form, variable=mMaritalStatus, width=30)
# entMaritalStatus.grid(row=1, column=1, padx=10, pady=10)

lblGender = Label(Form, text='Gender:')
lblGender.grid(row=2, column=0, padx=10, pady=10)

Gender = ['Male', 'Female']
mGender = StringVar()
for i in Gender:
    if i == 'Male':
        gender = ttk.Radiobutton(Form, variable=mGender, text=i, value=i).grid(row=2, column=1, padx=10, pady=10,
                                                                                                   sticky='w')
    else:
        gender = ttk.Radiobutton(Form, variable=mGender, text=i, value=i).grid(row=2, column=1, padx=10, pady=10,
                                                                                                   sticky='e')
# entGender = ttk.Entry(Form, textvariable=txtGender, width=30)
# entGender.grid(row=2, column=1, padx=10, pady=10)


lblEnglishEducation = Label(Form, text='EnglishEducation:')
lblEnglishEducation.grid(row=3, column=0, padx=10, pady=10)
EnglishEducation = ['Bachelors', 'GraduateDegree', 'HighSchool', 'PartialHighSchool', 'PartialCollege']

# txtEnglishEducation = StringVar()
entEnglishEducation = ttk.Combobox(Form, values=EnglishEducation, width=25)
entEnglishEducation.grid(row=3, column=1, padx=10, pady=10)



lblEnglishOccupation = Label(Form, text='EnglishOccupation:')
lblEnglishOccupation.grid(row=4, column=0, padx=10, pady=10)
EnglishOccupation = ['Clerical', 'Management', 'Manual', 'Professional', 'SkilledManual']
entEnglishOccupation = ttk.Combobox(Form, values=EnglishOccupation, width=25)
entEnglishOccupation.grid(row=4, column=1, padx=10, pady=10)


# lblDateFirstPurchase = Label(Form, text='DateFirstPurchase:')
# lblDateFirstPurchase.grid(row=2, column=4, padx=10, pady=10)
#
# # txtDateFirstPurchase = StringVar()
# entDateFirstPurchase = DateEntry(Form, width=25)
# entDateFirstPurchase.grid(row=2, column=5, padx=10, pady=10)

lblCommuteDistance = Label(Form, text='CommuteDistance:')
lblCommuteDistance.grid(row=3, column=4, padx=10, pady=10)

CommuteDistance = ['0-1Miles', '1-2Miles', '2-5Miles', '5-10Miles', '10+Miles']
entCommuteDistance = ttk.Combobox(Form, values=CommuteDistance, width=25)
entCommuteDistance.grid(row=3, column=5, padx=10, pady=10)

lblRegion = Label(Form, text='Region:')
lblRegion.grid(row=4, column=4, padx=10, pady=10)

Region = ['Pacific', 'NorthAmerica', 'Europe']
entRegion = ttk.Combobox(Form, values=Region, width=25)
entRegion.grid(row=4, column=5, padx=10, pady=10)

lblYearlyIncome = Label(Form, text='YearlyIncome:')
lblYearlyIncome.grid(row=5, column=4, padx=10, pady=10)

txtYearlyIncome = StringVar()
entYearlyIncome = ttk.Entry(Form, textvariable=txtYearlyIncome, width=25)
entYearlyIncome.grid(row=5, column=5, padx=10, pady=10)


lblTotalChildren = Label(Form, text='TotalChildren:')
lblTotalChildren.grid(row=5, column=0, padx=10, pady=10)

TotalChildren = [0, 1, 2, 3, 4, 5]
entTotalChildren = ttk.Combobox(Form, values=TotalChildren, width=25)
entTotalChildren.grid(row=5, column=1, padx=10, pady=10)

lblNumberChildrenAtHome = Label(Form, text='NumberChildrenAtHome:')
lblNumberChildrenAtHome.grid(row=6, column=4, padx=10, pady=10)

NumberChildrenAtHome = [0,1,2,3,4,5]
entNumberChildrenAtHome = ttk.Combobox(Form, values=NumberChildrenAtHome, width=25)
entNumberChildrenAtHome.grid(row=6, column=5, padx=10, pady=10)

lblHouseOwnerFlag = Label(Form, text='HouseOwnerFlag:')
lblHouseOwnerFlag.grid(row=6, column=0, padx=10, pady=10)

HouseOwner = [0,1]
entHouseOwnerFlag = ttk.Combobox(Form, values=HouseOwner, width=25)
entHouseOwnerFlag.grid(row=6, column=1, padx=10, pady=10)

lblNumberCarsOwned = Label(Form, text='NumberCarsOwned:')
lblNumberCarsOwned.grid(row=0, column=4, padx=10, pady=10)

NumberCarsOwned = [0, 1, 2, 3, 4]
entNumberCarsOwned = ttk.Combobox(Form, values=NumberCarsOwned, width=25)
entNumberCarsOwned.grid(row=0, column=5, padx=10, pady=10)

lblAge = Label(Form, text='Age:')
lblAge.grid(row=1, column=4, padx=10, pady=10)

entAge = ttk.Entry(Form, width=25)
entAge.grid(row=1, column=5, padx=10, pady=10)

btnPredict = ttk.Button(Form, width=15, text='Predict', command=check)
btnPredict.grid(row=16, column=5, padx=10, pady=10, sticky='e')


Form.mainloop()