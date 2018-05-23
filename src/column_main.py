
# coding: utf-8

# In[1]:


import numpy as np
import time
import matplotlib.pyplot as plt  


# In[2]:


def find_nearest_none_zero(mat,x_index,y_index):
    for y in range(y_index,mat.shape[1]):
        if(mat[x_index][y]!=0):
            return y
    return False
def find_abs_max(mat,x_index,y_index):
    temp_max = abs(mat[x_index,y_index])
    temp_index = y_index
    for y in range(y_index,mat.shape[1]):
        if(abs(mat[x_index][y]) > temp_max):
            temp_max = abs(mat[x_index][y])
            temp_index = y
    return y


# In[3]:


def guass_decrease_variable(mat,ans):
    mat = np.asarray(mat)
    ans = np.asarray(ans)
    if(len(mat.shape)!=2):
        print("wrong dimention")
        return
    else:
        if(mat.shape[0]!=mat.shape[1]):
            print("the x dimention doesnt equal to the y dimention")
            return
        else:
            for x in range(mat.shape[0]-1):
                if(mat[x][x]==0):
                        y_none_zero = find_nearest_none_zero(mat,x,x)
                        if(y_none_zero == False):
                            print('wrong det(mat)=0')
                            return False
                        mat[:,[x,y_none_zero]] = mat[:,[y_none_zero,x]]
                        (ans[x],ans[y_none_zero]) = (ans[y_none_zero],ans[x])
                for y in range(x+1,mat.shape[1]):
                    #if(mat[x][y]==0):
                    #if(mat[x][y]==0):
                    cover = mat[y][x] / mat[x][x]
                    #print(cover+99)
                    #print(cover)
                    mat[:][y] -= cover * mat[:][x]
                    #mat[x][y] = 0
                    ans[y] -= cover * ans[x]
            result = np.zeros([mat.shape[0]])
            for y in range(mat.shape[0]-1,-1,-1):
                result[y] = (ans[y] - np.sum((result[n]*mat[y][n]
                                              for n in range(y+1,mat.shape[0])))) / mat[y][y]
            return result


# In[4]:


def column_main_decrease_variable(mat,ans):
    mat = np.asarray(mat)
    ans = np.asarray(ans)
    if(len(mat.shape)!=2):
        print("wrong dimention")
        return
    else:
        if(mat.shape[0]!=mat.shape[1]):
            print("the x dimention doesnt equal to the y dimention")
            return
        else:
            for x in range(mat.shape[0]-1):
                if(mat[x][x]==0):
                        y_none_zero = find_abs_max(mat,x,x)
                        if(mat[x][x] == 0):
                            print('wrong det(mat)=0')
                            return False
                        mat[:,[x,y_none_zero]] = mat[:,[y_none_zero,x]]
                        (ans[x],ans[y_none_zero]) = (ans[y_none_zero],ans[x])
                for y in range(x+1,mat.shape[1]):
                    #if(mat[x][y]==0):
                    #if(mat[x][y]==0):
                    cover = mat[y][x] / mat[x][x]
                    #print(cover+99)
                    #print(cover)
                    mat[:][y] -= cover * mat[:][x]
                    #mat[x][y] = 0
                    ans[y] -= cover * ans[x]
            result = np.zeros([mat.shape[0]])
            for y in range(mat.shape[0]-1,-1,-1):
                result[y] = (ans[y] - np.sum((result[n]*mat[y][n]
                                              for n in range(y+1,mat.shape[0])))) / mat[y][y]
            return result


# In[6]:


if __name__ == '__main__':
    y_array = []
    x_array = [10,50,100,200]
    for x in (10,50,100,200):
        time_start = time.time()
        while(1):
            mat = np.random.normal(size=(x,x))
            ans = np.random.normal(size=(x))
            result = column_main_decrease_variable(mat,ans)
            if(not isinstance(result,bool)):
                #print
                res = result.reshape(-1,1).sort(axis=1)
                #print(sorted(result))
                #print(sorted(guass_decrease_variable(mat,ans)))
                #print(sorted(np.linalg.solve(mat,ans)))
                time_end = time.time()
                y_array.append(time_end-time_start)
                #print(time_start-time_end)
                break
    plt.scatter(x_array,y_array,color='red')
    plt.title('column main ')
    plt.ylabel('runtime (second)')
    plt.xlabel('number of dimention')
    plt.plot(x_array,y_array)
    '''
    mat = np.asarray([1,0,0,1],dtype=np.float).reshape(2,-1)
    ans = np.asarray([1,2],dtype=np.float)
    print(sorted(np.linalg.solve(mat,ans)))
    print(guass_decrease_variable(mat,ans))
    '''

