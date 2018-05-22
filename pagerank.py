
# coding: utf-8

# In[1]:


import numpy as np
#from scipy.sparse import dok_matrix
import pickle

import time
dims = 75879
a = np.full([dims],1)


# In[2]:


#x = np.asarray([1,2,1,1,11,1])
#np.where(x == 1)


# In[3]:


S = dok_matrix((dims+100,dims+100),dtype=bool)
#S[2,2] = 1
#print(S.toarray())
A = dok_matrix((dims,dims),dtype=float)
#A[0,0] = 1
#A[0,1] = 1 
#print(A.getrow(0).nonzero()[1])


# In[4]:


with open('soc.txt','r') as f:
    times = 0
    for line in f:
        if(len(line.split()) == 2):
            S[int(line.split()[0]),int(line.split()[1])] = True
            #print(line)
        times += 1
        if(times%10000 == 0):
            print(line)
f2 = open('S_LINK_MATRIX.txt','wb')
pickle.dump(S,f2)
f2.close()


# In[4]:


f2 = open('S_LINK_MATRIX.txt','rb')
S = pickle.load(f2)
#print(S)


# In[ ]:


time_start = time.time()
for y in range(dims):
    #sum_ = len(S.getrow(y).toarray())
    arr = np.asarray(S.getrow(y).nonzero()[1])
    
    arr_target = arr
    #print(arr)
    if(len(arr_target) != 0):
        for x in arr_target:
            A[x,y] = 1/len(arr_target)
    else:
        a[y] = 0
        #for x in range(dims):
            #A[x,y] = 1/dims
    if(y%10 == 0):
        time_end = time.time()
        print(time_end-time_start)
        time_start = time_end


# In[ ]:


apha = 0.85
def dot(mat1,v1):
    B = np.zeros(dims)
    for y in range(dims):
        arr = np.asarray(mat1.getrow(y).nonzero())
        sum_ = 0
        for index in arr:
            sum_ += arr[index,y] * v1[index]
        B[y] = sum_
    return B.reshape(-1,1)


# In[ ]:


def pagerank_iteration(A,iteration_times):
    p = np.zeros(dims).reshape(-1,1)
    for x in range(iteration_times):
        p = apha * (dot(A,p) + np.full([dims,1],1).dot(a.reshape(1,-1)).dot(v1) / dims ) + (1 - apha) * np.full([dims,1],np.sum(p)) / dims
    return p

