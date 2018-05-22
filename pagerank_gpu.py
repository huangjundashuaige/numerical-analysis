
# coding: utf-8

# In[7]:


import torch
import pickle
import time

import torch.nn as nn


# In[5]:





# In[6]:





# In[ ]:


class pagerank_gpu_pytorch():
    def __init__():
        self.dims = 75879
        self.apha = 0.85
        f2 = open('pytorch_linked_matrix.txt','w+')
        if(len(f2.read()) == 0):
            self.S = torch.zeros([self.dims,self.dims],dtype=torch.cuda.float).cuda()
            self.S = nn.DataParallel(self.S)
            with open('soc.txt','r') as f:
                times = 0
                for line in f:
                    if(len(line.split()) == 2):
                    S[int(line.split()[0])][int(line.split()[1])] = 1
                    #print(line)
                    times += 1
                    if(times%10000 == 0):
                        print(line)
                pickle.dump(S,f2)
                
        else:
            self.S = pickle.load(f2)
        f2.close()
    def transform_probility_matrix():
        for x in range(self.S.shape[0]):
            sum_ = torch.sum(S[x][:])
            if(sum_!=0);
                s[x][:] /= sum_
        self.S.transpose_(1,0)
    def pagerank_iteration(iteration_times):
        p = torch.from_numpy(np.zeros(dims).reshape(-1,1)).cuda()
        p = nn.DataParallel(p)
        #e = torch.from_numpy(np.full([dims,1],1)).cuda()
        #e = nn.DataParallel(e)
        #lack self transform
        for x in range(iteration_times):
            p = self.apha * (torch.dot(self.S,p) + 
                             (1 - self.apha) * torch.sum(p) / self.dims
        return p

