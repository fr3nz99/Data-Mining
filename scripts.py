# Hello_World

if __name__ == '__main__':
    print "Hello, World!"

#If_Else


import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(raw_input().strip())
if n%2!=0:
    print "Weird"
elif n<=5:
    print "Not Weird"
elif n<=20:
    print "Weird"
else:
    print "Not Weird"
    
#Arithmetic_Operators
if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a+b)
    print(a-b)
    print(a*b)
     
#Division
from __future__ import division

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
print(a//b)
print(a/b)

#Loops
if __name__ == '__main__':
    n = int(raw_input())
    for i in range(0,n):
        c=i*i
        print(c)
        
#Write_a_function
def is_leap(year):
    if year%4==0 and year%100!=0 or year%400==0 :
        return True
    else:
        return False

year = int(raw_input())
print is_leap(year)

#Print_function
from __future__ import print_function

if __name__ == '__main__':
    n = int(raw_input())

    
for i in range(1, n+1):
        print(i, end='')
    
    
    
    
    
    
    
    
    
 #DATA_TYPES

#List_comprehension
        if __name__ == '__main__':
    x = int(raw_input())
    y = int(raw_input())
    z = int(raw_input())
    n = int(raw_input())
    
    a=[]
    for i in range(x+1):
        for j in range(y+1):
            for k in range(z+1):
                if i+j+k!=n:
                    a.append([i,j,k])
print(a)
  
    
#Find_the_runnerup_score

if __name__ == '__main__':
    n = int(raw_input())
    arr = map(int, raw_input().split())

    c=max(arr)
    while max(arr)==c:
        arr.remove(c)
print(max(arr))



#Nested_Lists

n = int(raw_input())
lista = []
for x in range(0, n):
    lista.append([raw_input(), float(raw_input())])
lista = sorted(lista, key=lambda x: x[1]);
for x in range(1, n):
    if(lista[x][1] != lista[x-1][1]):
        voto = lista[x][1]
        break
lista = sorted(lista);
for x in range(n):
    if(lista[x][1] == voto):
        print lista[x][0]
        
  
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
