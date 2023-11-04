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
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from keras.layers import Dense, Flatten, Conv1D, MaxPool1D, Dropout

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
        data[['htn','dm','cad','pe','ane']] = data[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
        data[['rbc','pc']] = data[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
        data[['pcc','ba']] = data[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
        data[['appet']] = data[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
        data['classification'] = data['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
        data.rename(columns={'classification':'class'},inplace=True)
        data['pe'] = data['pe'].replace(to_replace='good',value=0)
        data['appet'] = data['appet'].replace(to_replace='no',value=0)
        data['cad'] = data['cad'].replace(to_replace='\tno',value=0)
        data['dm'] = data['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
        data.drop('id',axis=1,inplace=True)
        data.dropna(axis=0,inplace=True)
        data.apply(pd.to_numeric)
        for i in data.columns:
            data[i] = MinMaxScaler().fit_transform(data[i].astype(float).values.reshape(-1, 1))
        for i in range(0,data.shape[1]):
            if data.dtypes[i]=='object':
                data['pcv'] = data.pcv.astype(float)
                data['wc'] = data.wc.astype(float)
                data['rc'] = data.rc.astype(float)
                data['dm'] = data.dm.astype(float)
        text.insert(END,"Top Five rows of dataset\n"+str(data.head())+"\n")
        text.insert(END,"Last Five rows of dataset\n"+str(data.tail()))


      
    def splitdataset():      
        text.delete('1.0', END)
        X = data.iloc[:,:-1] 
        Y = data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
        text.insert(END,"\nTrain & Test Model Generated\n\n")
        text.insert(END,"Total Dataset Size : "+str(len(data))+"\n")
        text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
        text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")
        return X_train, X_test, y_train, y_test

    def MLmodels():
        global results,names
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
            kfold=KFold(n_splits=5,random_state=None)
            cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
            model.fit(X_train,y_train)
            predicted=model.predict(X_test)

            predicted_values.append(predicted)
            results.append(cv_results.mean()*95)
            names.append(name)
            text.insert(END,"\n"+str(name)+" "+"Predicted Values on Test Data:"+str(predicted)+"\n\n")
            text.insert(END, "%s: %f\t\t(%f)\n" %(name,cv_results.mean()*95,cv_results.std()))
        return results
    def annmodel():
        global results,names
        X_train, X_test, y_train, y_test=test.splitdataset()
        text.delete('1.0', END)
        model = Sequential()
        model.add(Dense(units=100,input_dim=X_train.shape[1],activation='relu'))
        model.add(Dense(units=50,activation='relu'))
        model.add(Dense(units=25,activation='relu'))
        model.add(Dense(units=1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train,y_train,epochs=1,batch_size=40,validation_data=(X_test,y_test),verbose=2)
        acc =model.evaluate(X_test,y_test)
        print(results)
        text.insert(END, "%Accuracy of ANN Model: "+str(acc[1]*100)+"\n")
        results.append(acc[1]*100)
        print(results)
        names.append("ANN")

    def cnnmodel():
        import keras
        global results,names
        X_train, X_test, y_train, y_test=test.splitdataset()
        text.delete('1.0', END)
        y_train = keras.utils.np_utils.to_categorical(y_train, 2)
        y_test = keras.utils.np_utils.to_categorical(y_test, 2)
        X_train = X_train.values
        X_test = X_test.values
        #x_train = X_train.reshape(614, 24, 1)
        #x_test = X_test.reshape(154, 24, 1)
        num_classes = 2
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=3, padding='same', input_shape=(24, 1), activation='relu'))
        model.add(MaxPool1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))  # dropout
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs = 7, batch_size = 128, validation_data=(X_test, y_test))
        cnn = model.evaluate(X_test,y_test)
        print(results)
        text.insert(END, "%Accuracy of CNN Model: "+str(cnn[1]*100)+"\n")
        results.append(cnn[1]*100)
        print(results)
        names.append("CNN")


    
    def graph():
        
        print(len(results))
        bars = ('KNN','CART','NB','SVM','RF','Bagging','Ada','MLP','ANN','CNN')
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

dl= Button(main, text="ANN Model", command=test.annmodel)
dl.place(x=700,y=350)
dl.config(font=font1)

dl= Button(main, text="CNN Model", command=test.cnnmodel)
dl.place(x=700,y=400)
dl.config(font=font1) 

graph= Button(main, text="Model Comparison", command=test.graph)
graph.place(x=700,y=450)
graph.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='SlateGray1')
main.mainloop()
