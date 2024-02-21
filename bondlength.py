import codecs
import numpy as np
from numpy import *

f = codecs.open('output.txt', mode='r', encoding='utf-8')  
line = f.readline()   
list1 = []

while line:
    a = line.split()
    b = a[3:4]   
    list1.append(b)  
    line = f.readline()
f.close()
mylist1 = np.array(list1)
mylist =mylist1.astype(float)
average = sum(mylist) / len(mylist)
#for i in list1:
   # print (i)
print("average:",average)

