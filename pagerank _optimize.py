
# coding: utf-8

# In[1]:


import numpy as np
from scipy.sparse import dok_matrix
import pickle

import time

dims = 75879+100
a = np.full([dims],1)


# In[2]:


x = np.asarray([1,2,1,1,11,1])
np.where(x == 1)


# In[3]:


S = dok_matrix((dims,dims),dtype=float)
occur_arr=dict()
occurence=dict()
#S[2,2] = 1
#print(S.toarray())
#A[0,0] = 1
#A[0,1] = 1 
#print(A.getrow(0).nonzero()[1])


# In[4]:


with open('soc.txt','r') as f:
    times = 0
    for line in f:
        if(len(line.split()) == 2):
            if(int(line.split()[0]) in occur_arr):
                occur_arr[int(line.split()[0])] += 1
                occurence[int(line.split()[0])].append(int(line.split()[1]))
            else:
                occur_arr[int(line.split()[0])] = 1
                occurence[int(line.split()[0])] = [int(line.split()[1])]
                #print(line)
        times += 1
        if(times%10000 == 0):
            print(line)


# In[5]:


print(occur_arr)


# In[6]:


with open('soc.txt','r') as f:
    times = 0
    for line in f:
        if(len(line.split()) == 2):
            if(int(line.split()[0]) in occur_arr):
                S[int(line.split()[0]),int(line.split()[1])] = 1/occur_arr[int(line.split()[0])]
            #print(line)
        times += 1
        if(times%10000 == 0):
            print(line)
f2 = open('S_LINK_MATRIX.txt','wb')
pickle.dump(S,f2)
f2.close()


# In[7]:


f2 = open('S_LINK_MATRIX.txt','rb')
S = pickle.load(f2)
#print(S)


# In[8]:


'''
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
'''
A = S.transpose()


# In[72]:


def dot2(v1):
    B = np.zeros(dims)
    for index in range(len(occur_arr)):
        if(occur_arr[index] != 0):
            _sum = 0
            k = 1 / occur_arr[index]
            for index2 in occurence.values():
                _sum += k * v1[index2][0]
            B[index] = _sum
    #print(_sum)
        if(index%100 == 0):
            print(time.time())
    return B.reshape(-1,1)


# In[73]:


apha = 0.85
def dot(mat1,v1):
    print(mat1.shape)
    print(v1.shape)
    B = np.zeros(dims)
    #v1 = v1.reshape(-1)
    for y in range(dims):
        arr = np.asarray(mat1.getrow(y).nonzero())
        sum_ = 0
        for index in arr:
            sum_ += mat1[index][y] * v1[index][0]
        B[y] = sum_
        #print(B.shape)
    return B.reshape(-1,1)


# In[74]:


def pagerank_iteration(A,iteration_times):
    p = np.zeros(dims).reshape(-1,1)
    for x in range(iteration_times):
        #p = apha * (dot2(p) + np.full([dims,1],1).dot(a.reshape(1,-1)).dot(v1) / dims ) + (1 - apha) * np.full([dims,1],np.sum(p)) / dims
        p = apha * (dot2(p)) + (1 - apha) * np.sum(p) / dims
        print(p)
        #p = dot2(p)#(1 - apha)) * np.full([dims,1],np.sum(p)) / dims
    return p


# In[75]:


with open('result.txt','wb') as f:
    res = pagerank_iteration(A,100)
    pickle.dump(res,f)


# In[80]:


v1 = np.zeros(dims).reshape(-1,1)
np.full([dims//10,1],1).dot(v1.transpose())

