import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#data is in mm. 


def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

#output indices of peaks
def find_local_maximums(arr, bounces, time_arr):
    local_maximums = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            local_maximums.append(arr[i])
            
    local_maximums = np.sort(local_maximums)[::-1]
    local_maximums = local_maximums[0:2*bounces-1]
    
    indices1 = np.zeros(len(local_maximums))
    for i in range(0, len(indices1)):
        indices1[i] = np.argwhere(arr==local_maximums[i]) 
    
    time_out = np.zeros(len(indices1))
    for i in range(0, len(time_out)):
        time_out[i] = time_arr[int(indices1[i])]
    
    return local_maximums
#     return indices1
#     return time_out

bounces = 4

general_array = pd.read_csv('newton.txt', sep=',')
general_array2 = pd.read_csv('newton_B.txt', sep=',')

t1 = general_array.values[:, 0] 
x1 = general_array.values[:, 1] 
y1 = general_array.values[:, 2] 

x1 = np.delete(x1,0)
y1 = np.delete(y1,0)
t1 = np.delete(t1,0)


x1 = np.array(x1).astype(float)*10**(-3)
y1 = np.array(y1).astype(float)*10**(-3)
t1 = np.array(t1).astype(float)

euclid1 = np.sqrt(x1**2 + y1**2)
fig = plt.figure(figsize=(4, 2))
plt.plot(t1, euclid1, 'b.-')
plt.title("1st ball position vs time")

########################################## get x component and plot all of them

t2 = general_array2.values[:, 0]
x2 = general_array2.values[:, 1] 
y2 = general_array2.values[:, 2] 

x2 = np.delete(x2,0)
y2 = np.delete(y2,0)
t2 = np.delete(t2,0)

x2 = np.array(x2).astype(float)*10**(-3)
y2 = np.array(y2).astype(float)*10**(-3)
t2 = np.array(t2).astype(float)

euclid2 = np.sqrt(x2**2 + y2**2)
fig = plt.figure(figsize=(4, 2))
plt.plot(t2, euclid2, 'r.-')
plt.title("2nd ball position vs time")

#get velocity 1
v1 = np.sqrt(np.gradient(euclid1, t1)**2)
#v1 = (np.gradient(euclid1, t1))
# fig = plt.figure(figsize=(4, 2))
# plt.plot(t1, v1, 'k.-')
# plt.title("1st ball velo vs time")

#get velocity 2
v2 = np.sqrt(np.gradient(euclid2, t2)**2)
#v2 = (np.gradient(euclid2, t2))
# fig = plt.figure(figsize=(4, 2))
# plt.plot(t2, v2, 'k.-')
# plt.title("2nd ball velo vs time")


# smoothing
window_size = 4
t1 = moving_average(t1, window_size)
v1 = moving_average(v1, window_size)
fig = plt.figure(figsize=(4, 2))
plt.plot(t1, v1, 'k.-')
plt.title("SMOOTHED 1st ball position vs time")

t2 = moving_average(t2, window_size)
v2 = moving_average(v2, window_size)
fig = plt.figure(figsize=(4, 2))
plt.plot(t2, v2, 'k.-')
plt.title("SMOOTHED 2nd ball position vs time")

m = 0.01 #kg
#get energy
e1 = 1/2*m*v1**2
e2 = 1/2*m*v2**2


a = find_local_maximums(e1, bounces, t1)
b = find_local_maximums(e2, bounces, t2)
energy_loss = (np.abs(a-b))
fig = plt.figure(figsize=(4, 2))
plt.plot(energy_loss, '--')
plt.plot(energy_loss, '.r', markersize = 15)
plt.ylabel("Energy lost (J)")
plt.xlabel("Collisions")
