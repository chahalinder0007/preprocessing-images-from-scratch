#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import depends
from preproces import preprocess
from sklearn import svm
from sklearn.metrics import confusion_matrix


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
#%matplotlib  #use this for interactive plots


# In[ ]:


X,Y=preprocess(path="/home/inder/freelanc/SAMPLE/").data(type_of_transform="dwt and dct")


# In[ ]:


classfier=svm.SVC(kernel="poly",degree=2) #prob is true only to examine values it slowes training though
d_len=int(len(X)*0.90) #95% of the data is segragated for training 5% for testing.

X_train=X[:d_len]
Y_train=Y[:d_len]
classfier.fit(X_train,Y_train)
#Y=[int(x) for x in Y]
print(classfier.score(X[d_len:],Y[d_len:]))        


# In[ ]:


#to save the classifier
import pickle
with open("classifierave.pickle","wb") as file:
    pickle.dump(classfier,file)


# In[ ]:


import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.rcParams["figure.figsize"]=15,10


# In[ ]:


proba_classi=classfier.predict_proba(X[d_len:])
x=[int(x) for x in Y[d_len:]]
y=[np.argmax(x) for x in proba_classi]
xy=np.vstack([x,y])
z=gaussian_kde(xy)(xy)

print(confusion_matrix(y_true=x,y_pred=y))


# In[ ]:


plt.scatter([int(x) for x in Y[d_len:]],[np.argmax(x) for x in proba_classi])
plt.show()


# In[ ]:


#print with density at points blue means least dense
fig,ax=plt.subplots()
ax.scatter(x,y,c=z,s=100,edgecolor="")
plt.show()


# In[ ]:


#density plot with reference bar
plt.hist2d(x,y,(50,50),cmap=plt.cm.jet)
plt.colorbar()
plt.show()

