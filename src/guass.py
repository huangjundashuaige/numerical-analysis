
# coding: utf-8

# In[2]:


import numpy as np
import time
import matplotlib.pyplot as plt  


# In[3]:



_x = np.full([2,2],1)
_x[1,:] = _x[1,:]+1
_x[1,:] -= 3

np.sum(x for x in range(100))
print(list(range(100,-1,-1)))


# In[4]:


def find_nearest_none_zero(mat,x_index,y_index):
    for y in range(y_index,mat.shape[1]):
        if(mat[x_index][y]!=0):
            return y
    return False


# In[5]:


def guass_decrease_variable(mat,ans):
    mat = np.asarray(mat,dtype=float)
    ans = np.asarray(ans,dtype=float)
    if(len(mat.shape)!=2):
        print("wrong dimention")
        return
    else:
        if(mat.shape[0]!=mat.shape[1]):
            print("the x dimention doesnt equal to the y dimention")
            return
        else:
            #print(mat)
            for x in range(mat.shape[0]-1):
                if(mat[x][x]==0):
                        y_none_zero = find_nearest_none_zero(mat,x,x)
                        if(y_none_zero == False):
                            print('wrong det(mat)=0')
                            return False
                        (mat[:,x],mat[:,y_none_zero]) = (mat[:,y_none_zero],mat[:,x])
                        (ans[x],ans[y_none_zero]) = (ans[y_none_zero],ans[x])
                for y in range(x+1,mat.shape[1]):
                    #if(mat[x][y]==0):
                    cover = mat[y][x] / mat[x][x]
                    #print(cover+99)
                    #print(cover)
                    mat[:][y] -= cover * mat[:][x]
                    #mat[x][y] = 0
                    ans[y] -= cover * ans[x]
                #print(mat[x,:])
            result = np.zeros(mat.shape[0])
            #print(result)
            #print(mat)
            #print(mat[1][0]+100)
            for y in range(mat.shape[0]-1,-1,-1):    
                result[y] = (ans[y] - np.sum((result[n]*mat[y][n]
                                              for n in range(y+1,len(result))))) / mat[y][y]
                #print(result)
                #print(result)
            return result


# In[7]:


if __name__ == '__main__':
    y_array = []
    x_array = [10,50,100,200]
    for x in (10,50,100,200):
        time_start = time.time()
        while(1):
            mat = np.random.normal(size=(x,x))
            ans = np.random.normal(size=(x))
            #mat = np.asarray([1,3,2,1]).reshape(2,-1)
            #ans = np.arange(2)
            result = guass_decrease_variable(mat,ans)
            if(not isinstance(result,bool)):
                #print
                res = result.reshape(-1,1).sort(axis=1)
                print(sorted(result))
                print(sorted(np.linalg.solve(mat,ans)))
                time_end = time.time()
                y_array.append(time_end-time_start)
                #print(time_start-time_end)
                break
    plt.scatter(x_array,y_array,color='red')
    plt.title('guass ')
    plt.ylabel('runtime (second)')
    plt.xlabel('number of dimention')
    plt.plot(x_array,y_array)
    '''
    mat = np.asarray([1,0,0,1],dtype=np.float).reshape(2,-1)
    ans = np.asarray([1,2],dtype=np.float)
    print(sorted(np.linalg.solve(mat,ans)))
    print(guass_decrease_variable(mat,ans))
    '''

