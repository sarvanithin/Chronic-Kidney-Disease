from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
from tkinter.filedialog import askopenfilename

import numpy as np 
import pandas as pd 


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


import pylab as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
pl.style.use('fivethirtyeight')
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

main = tkinter.Tk()
main.title(" CHRONIC KIDNEY DISEASE ANALYSIS USING DATA MINING CLASSIFICATION TECHNIQUES")
main.geometry("1300x1200")

class test:
    def upload():
        global filename
        text.delete('1.0', END)
        filename = askopenfilename(initialdir = "Dataset")
        pathlabel.config(text=filename)
        text.insert(END,"Dataset loaded\n\n")

    def csv():
        global data
        text.delete('1.0', END)
        data=pd.read_csv(filename)
        print(data.dtypes)
        for i in ['rc','wc','pcv']:
                data[i] = data[i].str.extract('(\d+)').astype(float)
                for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:
                        print(data[i].mean(),i)
                        data[i].fillna(data[i].mean(),inplace=True)
        data['dm'] = data['dm'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})
        data['dm'].fillna('no',inplace=True)
        data['pc'].fillna('normal',inplace=True)
        data['rbc'].fillna('normal',inplace=True)
        data['ba'].fillna('notpresent',inplace=True)
        data['pcc'].fillna('notpresent',inplace=True)
        data['htn'].fillna('no',inplace=True)
        data['cad'] = data['cad'].replace(to_replace='\tno',value='no')
        data['cad'].fillna('no',inplace=True)
        data['appet'].fillna('good',inplace=True)
        data['pe'].fillna('no',inplace=True)
        data['ane'].fillna('no',inplace=True)
        data['cad'] = data['cad'].replace(to_replace='ckd\t',value='ckd')
        for i in ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','classification']:
                data[i] = LabelEncoder().fit_transform(data[i])
        for i in data.columns:
                data[i] = MinMaxScaler().fit_transform(data[i].astype(float).values.reshape(-1, 1))
        text.insert(END,"Top Five rows of dataset\n"+str(data.head())+"\n")
        text.insert(END,"Last Five rows of dataset\n"+str(data.tail()))


        
    def splitdataset():      
        text.delete('1.0', END)
        X = data.iloc[:,:-1] 
        Y = data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 7)
        text.insert(END,"\nTrain & Test Model Generated\n\n")
        text.insert(END,"Total Dataset Size : "+str(len(data))+"\n")
        text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
        text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")
        return X_train, X_test, y_train, y_test

    def MLmodels():
        X_train, X_test, y_train, y_test=test.splitdataset()
        text.delete('1.0', END)
        models=[]      
        models.append(('KNN',KNeighborsClassifier()))
        models.append(('CART',DecisionTreeClassifier()))
        models.append(('NB',GaussianNB()))
        models.append(('SVM',SVC()))
        models.append(('RF',RandomForestClassifier()))
        models.append(('Bagging',BaggingClassifier()))
        models.append(('Ada',AdaBoostClassifier()))
        models.append(('MLP',MLPClassifier()))
        results=[]
        names=[]
        predicted_values=[]
        text.insert(END,"Machine Learning Classification Models\n")
        text.insert(END,"Predicted values,Accuracy Scores and S.D values from ML Classifiers\n\n")
        for name,model in models:
            kfold=KFold(n_splits=10,random_state=7)
            cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
            model.fit(X_train,y_train)
            predicted=model.predict(X_test)

            predicted_values.append(predicted)
            results.append(cv_results.mean()*100)
            names.append(name)
            text.insert(END,"\n"+str(name)+" "+"Predicted Values on Test Data:"+str(predicted)+"\n\n")
            text.insert(END, "%s: %f\t\t(%f)\n" %(name,cv_results.mean()*100,cv_results.std()))
        return results
            
    def graph():
        results=test.MLmodels()     
        bars = ('KNN','CART','NB','SVM','RF','Bagging','Ada','MLP')
        y_pos = np.arange(len(bars))
        plt.bar(y_pos, results)
        plt.xticks(y_pos, bars)
        plt.show()

    

    




font = ('times', 16, 'bold')
title = Label(main, text=' CHRONIC KIDNEY DISEASE ANALYSIS USING DATA MINING CLASSIFICATION TECHNIQUES')
title.config(bg='sky blue', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=test.upload)
upload.place(x=700,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

df = Button(main, text="Reading Data ", command=test.csv)
df.place(x=700,y=200)
df.config(font=font1)

split = Button(main, text="Train_Test_Split ", command=test.splitdataset)
split.place(x=700,y=250)
split.config(font=font1)

ml= Button(main, text="All Classifiers", command=test.MLmodels)
ml.place(x=700,y=300)
ml.config(font=font1) 

graph= Button(main, text="Model Comparison", command=test.graph)
graph.place(x=700,y=350)
graph.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='SlateGray1')
main.mainloop()
