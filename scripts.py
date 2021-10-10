#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Done 7/7

# In[1]:


#Say_Hello_World
if __name__ == '__main__':
    print("Hello, World!")


# In[ ]:


#Python_if_else
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
if n%2!=0:
    print("Weird")
elif n<=5:
    print ("Not Weird")
elif n<=20:
    print ("Weird")
else:
    print ("Not Weird")


# In[ ]:


#Aritmetic_operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)


# In[ ]:


#Python_Division
from __future__ import division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)


# In[ ]:


#Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(0,n):
        c=i*i
        print(c)


# In[ ]:


#Write_a_function
def is_leap(year):
    if year%4==0 and year%100!=0 or year%400==0 :
        return True
    else:
        return False

year = int(input())
print(is_leap(year))


# In[ ]:


#Print_Function
if __name__ == '__main__':
    n = int(input())

for i in range(1, n+1):
        print(i, end='')


# # Basic Data Types

# Done 5/6, 1 copied from the web

# In[ ]:


#List_comprehension
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    a=[]
    for i in range(x+1):
        for j in range(y+1):
            for k in range(z+1):
                if i+j+k!=n:
                    a.append([i,j,k])
print(a)


# In[ ]:


#Find_the_RunnerUp_Score
if __name__ == '__main__':
    n = int(input())
    
    arrei = map(int, input().split())
    print (sorted(set(arrei))[-2]) #by using the [-2], I'm taking the penultimate element of the sorted array


# In[ ]:


#Nested_List
if __name__ == '__main__':
    
    tab = []
    punteggi = set()
    n=int(input())
for i in range(n):
    name = input()
    score = float(input())
    tab.append([name, score])
    punteggi.add(score)# putting the scores into a set
        
    secondoforte = []
for x, score in tab:
    secondodebole = sorted(punteggi)[1]
    if score == secondodebole:
        secondoforte.append(x)#if the score is the second lowest (so it's in the position 1), I print it.
        #By the == condition I'm assuming that, if there are more than one second lowest, 
        #I'm gonna print everyone of them

for x in sorted(secondoforte):
    print(x, end='\n')


# In[ ]:


#Finding_The_Percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    #The exercise requires only the next code-line
    print( format(sum(student_marks[query_name])/len(student_marks[query_name]), ".2f") )
    #With the .2f command I specify 2 digits of precision and f means floating point number


# In[ ]:


#Lists
if __name__ == '__main__':
    L = []
for _ in range(0, int(input())):
    user_input = input().split(' ')
    command = user_input.pop(0)
    if len(user_input) > 0:
        if 'insert' == command:
            eval("L.{0}({1}, {2})".format(command, user_input[0], user_input[1]))
        else:
            eval("L.{0}({1})".format(command, user_input[0]))
    elif command == 'print':
        print(L)
    else:
        eval("L.{0}()".format(command))
#copied from the web


# In[ ]:


#Tuples
if __name__ == '__main__':
    tupla=()
    n = int(input())
    integer_list = map(int, input().split())
    
tupla=tuple(integer_list) #Using the hash command suggested by hackerrank
print(hash(tupla))


# # Strings

# Done 11/14, 1 copied, 2 not found

# In[ ]:


#sWAP_cASE
def swap_case(s):
    parole = ''
    for i in s:
        if i == i.upper():
            parole = parole + i.lower()# I'm substituting every uppercase letter with an uppercase
        else:
            parole = parole + i.upper()# I'm substituting every lowercase letter with a lowercase
    return parole
#------------------------------------
if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


# In[ ]:


#String_Split_and_Join
def split_and_join(line):
    line = line.split(" ")
    line = "-".join(line)
    return line
#--------------------------------------
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# In[ ]:


#Whats_Your_Name
def print_full_name(first, last):
    print ('Hello', first, last+"! You just delved into python.")
#-------------------------------------
if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


# In[ ]:


#Mutations
def mutate_string(string, position, character):
    parolone=  string[:position] + character + string[position+1:]
    return parolone
#------------------------------------
if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


# In[ ]:


#Find_a_String
def count_substring(string, sub_string):
    a = 0
    for i in range(len(string)):
        if string[i:].startswith(sub_string):
            a = a + 1
    return a
#---------------------------------------
if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)


# In[ ]:


#String_Validators
if __name__ == '__main__':
    inp = input()
    print(any(c.isalnum() for c in inp))
    print(any(c.isalpha() for c in inp))
    print(any(c.isdigit() for c in inp))
    print(any(c.islower() for c in inp))
    print(any(c.isupper() for c in inp))


# In[ ]:


#Text_Wrap
import textwrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


# In[ ]:


#Designer_Door_Mat
N, M = map(int,input().split()) 
for i in range(1,N,2): 
    print(('.|.'*i).center(M,'-'))
print ("WELCOME".center(M,'-'))
for i in range(N-2,-1,-2): 
    print(('.|.'*i).center(M,'-'))
    #Entirely copied by the web


# In[ ]:


#Text_Alignment

#So impossible that I can't find a solution on the web


# In[ ]:


#String_Formatting
def print_formatted(n):
    cicci = len("{0:b}".format(n))
    for i in xrange(1,n+1):
        print "{0:{cicci}d} {0:{cicci}o} {0:{cicci}X} {0:{cicci}b}".format(i, cicci=cicci)
#---------------------------
if __name__ == '__main__':
    n = int(raw_input())
    print_formatted(n)


# In[ ]:


#Alphabet_Rangoli

#Impossible


# In[ ]:


#Capitalize
#!/bin/python3

import math
import os
import random
import re
import sys
#---------------------------
def solve(s):
    for i in s[:].split():
        s = s.replace(i, i.capitalize())
    return(s)
#replacing every first letter with a capitalized letter using replace command
#---------------------------
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()


# In[ ]:


#The_Minion_Game
def minion_game(string):
    vocali='AEIOU'
    Stuart=0
    Kevin=0
    for i in range(len(string)):
        if string[i] in vocali:
            Kevin=Kevin+len(string)-i#increasing Kevin's score by adding points for every vocal-word we can create 
        else:
            Stuart=Stuart+len(string)-i#increasing Stuart's score by adding points for every consonant-word we can create 
    if Kevin>Stuart:
        print ('Kevin', Kevin)
    elif Kevin<Stuart:
        print ('Stuart', Stuart)
    else:
        print ('Draw')
#------------------------------------
if __name__ == '__main__':
    s = input()
    minion_game(s)


# In[ ]:


#Merge_the_tools
def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        lettere = ''
        
        for i in string[i : i+k]:
            if (i not in lettere):
                lettere=lettere+i
        print(lettere)
#---------------------------------
if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)


# # Sets

# Done 10/13, 3 copied

# In[ ]:


#Introduction_to_Sets
def average(array):
    return sum(set(array))/len(set(array))
#-------------------------------
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)


# In[ ]:


#No_Idea
n, m = input().split()
ass = input().split()
A = set(input().split())
B = set(input().split())

quano=0
for i in ass:
    if i in A:
        quano=quano+1
    if i in B:
        quano=quano-1
print(quano)


# In[ ]:


#Symmetric_Difference
num , numvero = (int(input()),input().split())
num2 , numvero2=(int(input()),input().split())

numvero=set(numvero)
numvero2=set(numvero2)

diff1=numvero.difference(numvero2)
diff2=numvero2.difference(numvero)

tot=diff1.union(diff2)
print ('\n'.join(sorted(tot, key=int)))# printing the sorted symmetric difference created using the join command


# In[ ]:


#Set_add
sguan=set([str(input()) for i in range(int(input()))])
#By using the set() command, we are generating a set composed by unique attributes
#Now it's easy to count the single elements in the input
print(len(sguan))


# In[ ]:


#Set_Discard_Remove_Pop
n = int(input())
s = set(map(int, input().split()))

for i in range(int(input())):
    eval('s.{0}({1})'.format(*input().split()+['']))
print(sum(s))


# In[ ]:


#Set_Union_operation
n, ins1, m, ins2  = input(), set(input().split()), input(), set(input().split())
print(len(ins1.union(ins2)))


# In[ ]:


#Set_Intersection_operator
numero1, studenti1, numero2, studenti2 = (set(input().split()) for i in range(4))
print(len(studenti1.intersection(studenti2)))


# In[ ]:


#Set_Difference_operator
numero1, studenti1, numero2, studenti2 = (set(input().split()) for i in range(4))
print (len(studenti1.difference(studenti2)))


# In[ ]:


#Set_Symmetric_difference_operator
n, ins1, m, ins2= input(), set(input().split()), input(), set(input().split())
print(len(ins1.symmetric_difference(ins2)))


# In[ ]:


#The_Captains_room
k, stanze  = input(), input().split() 
capitano, multiple = set(), set()
for n in stanze:  
    if n not in capitano:
         capitano.add(n) 
    else:
         multiple.add(n)

print(capitano.difference(multiple).pop())


# In[ ]:


#Check_Subset
for i in range (int(input())):
        _, a = input(), set(input().split())
        _, b = input(), set(input().split())
        print(b.intersection(a) == a)
#Copied from the web


# In[ ]:


#Check_Strict_Superset
insiemone=set(input().split())
n=int(input())

sonosuperset=[]
for i in range(n):
     sonosuperset.append(insiemone.issuperset(set(map(str, input().split(' ')))))
     trueorfalse=all(sonosuperset)
print(trueorfalse)
#Copied from the web


# # Collections

# Done 

# In[ ]:


#Collections_counter
n=int(input())
scarpe=list(map(int,input().split()))
earn=0
ncostumers=int(input())

for i in range(ncostumers):
    taglia, prezzo=map(int,input().split())
    
    if taglia not in scarpe:
        continue    #If the size is not in the warehouse, the algorithm will continue looking for other sizes.
    else:
        earn=prezzo+earn     #If the size is found, the algorithm increments the profit of Raghu
        scarpe.remove(taglia)#Then the algorithm removes the size from the warehouse 
print(earn)


# In[ ]:


#DeafultDict_tutorial
from collections import defaultdict
d = defaultdict(list)
n,m = map(int,input().split())

for i in range(1,n+1):
    d[input()].append(i)
    
for j in range(1,m+1):
    gg = input()
    if gg in d:
        print(' '.join(map(str,d[gg])))
    else:
        print(-1)


# In[ ]:


#Collections_namedtuple
from collections import namedtuple

n = int(input())
nomi = input().split()
total = 0
for i in range(n):
    students = namedtuple('Ragazzi',nomi)
    Mark, Class, Name, Id = input().split()
    student = students(Mark,Class,Name,Id)
    total = total +int(student.MARKS)
print('{:.2f}'.format(total/n))


# In[ ]:


#Collections_orderedDict
lista = dict()
for i in range(int(input())):
    key, x, prezzo = input().rpartition(" ")
    lista[key] = lista.get(key,0) + int(prezzo)
for oggetto ,prezzo in lista.items():
    print(oggetto, prezzo)


# In[ ]:


#Word_Order
n = int(input())
cardinalita = {}


for i in range(0,n):
    parola= input()
    if parola in cardinalita:
        cardinalita[parola]=cardinalita[parola] + 1
    else:
        cardinalita[parola] = 1
print(len(cardinalita))
for i in cardinalita:
    print(cardinalita[i], end=" ")


# In[ ]:


#Collections_deque
from collections import deque
queue = deque()
for i in range(int(input())):
    inp = input().split()
    if inp[0]=="append":
        queue.append(inp[1])
    elif inp[0]=="appendleft":
        queue.appendleft(inp[1])
    elif inp[0]=="pop":
        queue.pop()
    elif inp[0]=="popleft":
        queue.popleft()
print(*queue)


# In[ ]:


#Pilling_up
from collections import deque

for _ in range(int(input())):  
    _, queue =input(), deque(map(int, input().split()))
    
    for cube in reversed(sorted(queue)):
        if queue[-1] == cube: queue.pop()
        elif queue[0] == cube: queue.popleft()
        else:
            print('No')
            break
    else: print('Yes')
    
    #copied from discussions


# In[ ]:


#Company_logo
import math
import os
import random
import re
import sys
from collections import Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
    pass
[print(*c) for c in OrderedCounter(sorted(input())).most_common(3)]

#copied from discussions


# ## Date and Time

# In[ ]:


#Calendar_module
import calendar
giorno=(input().split())
month=int(giorno[0])
day=int(giorno[1])
year=int(giorno[2])
numgiorno=calendar.weekday(year, month, day)
settimana=['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
print(settimana[numgiorno])#Printing the correspondent day of the week given by the function calendar.weekday


# In[ ]:


#Time_Delta
from datetime import datetime as dt


fmt = '%a %d %b %Y %H:%M:%S %z'
for i in range(int(input())):
    print(int(abs((dt.strptime(input(), fmt) - 
                   dt.strptime(input(), fmt)).total_seconds())))
                   
#copied from the web


# ## Errors and Exceptions

# In[ ]:


#Exceptions
n=int(input())
for i in range(0,n):
    try:
        num1, num2=map(int, input().split())
        print(num1//num2)
    except Exception as e:
        print("Error Code:",e)


# # Built-Ins

# In[1]:


#Zipped!
n, m = map(int, input().split())
saro=[]
for i in range(0,m):
    saro.append(map(float, input().split()))

sarrus=zip(*saro)
for i in sarrus:
    media=(sum(i)/len(i))
    print(media)


# In[ ]:


#Athlete_sort
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    arr = [list(map(int, input().split())) for i in range(n)]
    k = int(input())

    arr.sort(key=lambda x: x[k]) 
    #sorting the elements of the array arr by the k_th column

    for line in arr:
        print(*line, sep=' ')


# In[ ]:


#ginortS
def chiave(x):
    if x.islower():
        return(1,x)
    elif x.isupper():
        return(2,x)
    elif x.isdigit() :
        if int(x)%2==1:
            return(3,x)
        else :
            return(4,x)

print(*sorted(input(),key=chiave),sep='')
#the idea has been taken from the discussions. The question wasn't clear


# # Python Functionals

# In[ ]:


#Map_and_Lambda
cube = lambda x: pow(x,3)

def fibonacci(n):
    lista=[0,1]
    if n==0:
        return []
    if n==1:
        return [0]
    else:
        for i in range(2,n):
            lista.append(lista[i-2] + lista[i-1])
        return(lista[0:n])
#i don't know why, but hackerrank wants that for input 0 and 1 
#the output must be [] and [0], so I gave them those two values


# # Regex and Parsing

# In[ ]:


#Detect_Floating_Point_Number
import re
for _ in range(int(input())):
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', input())))
#copied from the web


# In[ ]:


#Re_split
regex_pattern = r"[,,.]" # , and . are the characters that divide the number


# In[ ]:


#Group_Groups_Groupdict
import re
out = re.search(r'([a-zA-Z0-9])\1+', input().strip())
print(out.group(1) if out else -1)


# In[ ]:


#Re_Findall_re_finditer
import re
consonanti = '[qwrtypsdfghjklzxcvbnm]'
out = re.findall('(?<=' + consonanti +')([aeiou]{2,})' + consonanti, input(), re.I)

print('\n'.join(out or ['-1']))
#copied from the web


# In[ ]:


#Re_start_Re_end
import re

a, b = input(), input()
if b in a:
    print(*[(i.start(), (i.start()+len(b)-1)) for i in re.finditer(r'(?={})'.format(b), a)], sep='\n')
    #for every element I look for the start index (i.start) and the end index (i.start+len(b)-1), 
    #given by the i.start+ the lenght of the array till the last index
else:
    print('(-1, -1)')


# In[ ]:


#Regex_substitution
import re
n = int(input())

for i in range(n):
    print(re.sub(r'(?<= )(&&|\|\|)(?= )', lambda x: 'and' if x.group() == '&&' else 'or', input()))
#copied from the web


# In[ ]:


#Validating_roman_numbers
regex_pattern = r"M{,3}(CM|CD|D?C{,3})(XC|XL|L?X{,3})(IX|IV|V?I{,3})$""


# In[ ]:


#Validating_phone_numbers
import re
for _ in range(int(input())):
  n = input()
  mec = re.match('^[789][0-9]{9}$', n)
  #re.match search the regular expression pattern and return the first occurrence, 
  #so if the number starts with 7-8-9, I will print YES
  if mec:
    print('YES')
  else:
    print('NO')


# In[ ]:


#Validating_and_parsing_email_addresses
import re
import email.utils

for i in range(int(input())):
    nome, mail = map(str,email.utils.parseaddr(input()))
    #i use parseaddr to return a tuple of the input
    emails = re.search(r'^[a-zA-Z]+[a-zA-Z0-9_.-]+[@][a-zA-Z]+[.][a-zA-Z]{1,3}$', mail)
    #I'm scanning through the string looking for the first location where the expression pattern produces a match
    if emails:
        print(email.utils.formataddr((nome, mail)))


# In[ ]:


#Hex_Color_code
import re

ricerca=r'(#[0-9a-fA-F]{3,6}){1,2}[^\n ]'
for _ in range(int(input())):
    for x in re.findall(ricerca,input()):
        print(x)


# In[ ]:


#Html_parser_part1
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        if attrs:
            for val in attrs:
                print("->",val[0],">",val[1])
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        if attrs:
            for val in attrs:
                print("->",val[0],">",val[1])
                
parser = MyHTMLParser()
n = int(input())
if n >= 1:
    res = ''.join(input().strip() for _ in range(n))
parser.feed(res)

#copied from discussions


# In[ ]:


#HTML_parser_part2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print(">>> Multi-line Comment")
            print(data)
        else:
            print(">>> Single-line Comment")
            print(data)
            
    def handle_data(self, data):
        if data != '\n':
            print(">>> Data")
            print(data)

            
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()
#copied from the web


# In[ ]:


#Detect_Html_tags_attributes
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self,gg,attributi):
        print(gg)
        for i in attributi:
            print('->',i[0],'>',i[1])
parser=MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())     


# In[ ]:


#Validating_UID
import re

#let's define the rules
almeno2 = r"(?=(?:.*[A-Z]){2,})"
almeno3num = r"(?=(?:.*\d){3,})"
soloalfanumerici = r"[a-zA-Z0-9]{10}"
noripetizioni = r"(?!.*(.).*\1)"
filtri = almeno2, almeno3num, soloalfanumerici, noripetizioni


for i in [input() for _ in range(int(input()))]:
    
    if all([re.match(f, i) for f in filtri]) and len(i)==10: 
        #we have to add the rule that there should be exactly 10 characters
        print("Valid")
    else:
        print("Invalid")


# In[ ]:


#Validating_credit_card_numbers
import re

p1 = r'^([456]\d{3})(\-?)(\d{4})(\2)(\d{4})(\2)(\d{4})$'
p2 = r'^(?:([0-9])(?!\1{3,})){16}$'
#The previous two lines are taken from the web
n=int(input())

for i in range(0,n):
    s = input()
    if re.match(p1, s):
        s = s.replace('-', '')
        if re.match(p2, s):
            print('Valid')
        else:
            print('Invalid')
    else:
        print('Invalid')


# In[ ]:


#Validating_postal_code

#we can clearly remove the previous rows and replace them with a bool function 
import re

n=input()

print (bool(re.match(r'^[1-9][\d]{5}$',n) and len(re.findall(r'(\d)(?=\d\1)',n))<2 ))


# In[ ]:


import math
import os
import random
import re
import sys


nm = input().split()
n = int(nm[0])
m = int(nm[1])
matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
string = ""      
for i in range(m):
    string += "".join(matrix[j][i] for j in range(n))
res = re.sub(r"(?<=\w)[!@#$%&\s]+(?=\w)",r" ",string)    
sys.stdout.write(res)
#Copied from the web


# # XML

# In[ ]:


#XML_1
def get_attr_number(node):
    return sum(len(child.attrib) for child in node.iter())
#copied


# In[ ]:


#XML_2
maxdepth = 0
def depth(elem, level):
    global maxdepth
    
    if (level == maxdepth):
        maxdepth = maxdepth + 1
    
    for child in elem:
        depth(child, level + 1)
        #for every corrispondence between the level and the maximum depth, maxdepth increases
#copied


# # Closures and Decorations

# In[ ]:


#Standardize_mobile_numbers
def wrapper(f):
    def fun(l):
        fun=f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
        #I'm gonna use c[-10:-5], c[-5:] instead of using c[0:5] c[5:] 
        #because the first method deletes the exceeded zeros, 
        #and the second tend to conserve the zeros on the edge
    return fun


# In[ ]:


#Decorators_2
def age(persona):
    eta=int(persona[2])
    return eta

def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=age))
    return inner


# # Numpy

# In[ ]:


#Arrays
def arrays(arr):
    arrei = (numpy.array(arr,float))
    #I'm gonna transfer the numbers into a float array, then i reverse it with the next command
    return arrei[::-1]


# In[ ]:


#Shape_and_reshape
import numpy
gg=numpy.array(input().split(),int)
gg.shape=(3,3)
print(gg)


# In[1]:


#Transpose_and_flatten
import numpy

n, m=map(int, input().split())

array = numpy.array([input().strip().split() for i in range(n)], int)

print(numpy.transpose(array))
print(array.flatten())


# In[ ]:


#Concatenate
import numpy

n, m, p=map(int, input().split())
c=(n+m)*p
array1 = numpy.array([input().split() for i in range(n)],int)
array2 = numpy.array([input().split() for i in range(m)],int)
print(numpy.concatenate((array1, array2), axis = 0))


# In[ ]:


#Zeros_and_ones
import numpy

n = list(map(int, input().split()))
print (numpy.zeros(n, dtype = numpy.int))
print (numpy.ones(n, dtype = numpy.int))


# In[ ]:


#Eye_and_identity
import numpy

numpy.set_printoptions(sign=' ') 
#this line is taken from the web, because the matrix printed from the next line has only one space ' ',
#but hackerrank requires two spacebars 
n,m=map(int,input().split())
print(numpy.eye(n,m))


# In[ ]:


#Array_mathematics
import numpy

n, m = map(int, input().split())
A, B = (numpy.array([input().split() for i in range(n)], dtype=int) for i in range(2))
print(A+B)
print(A-B)
print(A*B)
print(A//B)
print(A%B)
print(A**B)


# In[ ]:


#Floor_Ceil_Rint
import numpy
numpy.set_printoptions(sign=' ') 
#I found this function on the web. 
#It separates the elements in the print function by the sign

a = numpy.array(input().split(), float)
print(*(f(a) for f in [numpy.floor, numpy.ceil, numpy.rint]), sep='\n')


# In[ ]:


#Sum_and_prod
import numpy

N, M= map(int,input().split())
arrei = numpy.array([input().split() for i in range(M)], int)
print(numpy.prod((numpy.sum(arrei,axis=0))))


# In[ ]:


#Min_and_max
import numpy

arrei = []
n, m=map(int,input().split())

for i in range(n):
    inp=list(map(int,input().split()))
    arrei.append(inp)
    
my_array=numpy.array(arrei)
result=numpy.min(my_array, axis = 1)

print(max(result))


# In[ ]:


#Mean_var_std
import numpy

arrei = []
n, m = map(int, input().split())
for i in range(n): 
    arrei.append(list(map(int, input().split())))
    #just applying the previous functions
arrei = numpy.array(arrei)
print(numpy.mean(arrei, axis=1))
print(numpy.var(arrei, axis=0))
print(round(numpy.std(arrei), 11))


# In[ ]:


#Dot_and_cross
import numpy
 
n = int(input())
a = numpy.array([input().split() for i in range(n)], int)
b = numpy.array([input().split() for i in range(n)], int)
print(numpy.dot(a, b))


# In[ ]:


#Inner_and_outer
import numpy

A, B =numpy.array(input().split(),int), numpy.array(input().split(),int)

print(numpy.inner(A,B))
print(numpy.outer(A,B))


# In[ ]:


#Polynomials
import numpy

n = list(map(float,input().split()))
m = int(input())
print(numpy.polyval(n,m))


# In[ ]:


#Linear_algebra
import numpy

n = int(input())
arr = numpy.array([input().split() for _ in range(n)], float)
arr_det = numpy.linalg.det(arr)
#taken from the web
print(round(arr_det, 2))


# # Problems

# In[ ]:


#Birthday_Cake_Candles
import math
import os
import random
import re
import sys

#--------------------------------------
def birthdayCakeCandles(candles):
    candele=list(candles)
    maximo=max(candele)
    n=candele.count(maximo)
    return n
#--------------------------------------

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


# In[ ]:


#Number_Line_Jumps
import math
import os
import random
import re
import sys

#--------------------------------------
def kangaroo(x1, v1, x2, v2):
    kangaroo1=x1
    kangaroo2=x2
    c=0
    for x in range(0,10000):
        kangaroo1=kangaroo1+v1
        kangaroo2=kangaroo2+v2
        if kangaroo1==kangaroo2:
            c=1
    if c==1 : 
        return('YES')
    else:
        return('NO')
#--------------------------------------   

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()


# In[ ]:


#Viral_Advertising
import math
import os
import random
import re
import sys

#--------------------------------------
def viralAdvertising(n):
    c=2
    tot=2
    for i in range(n-1):
        c=((c*3)//2)
        tot= tot+c
    return tot
#--------------------------------------

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()


# In[ ]:


#Recursive_Digit_sum
import math
import os
import random
import re
import sys

#------------------------------------
def superDigit(n, k):
    n=str(n)
    nums=map(int, n)
    lista=list(nums)
    c=0
    for i in lista:
        c=i*k+c
    if c<10 :
        return c
    else: 
        return(superDigit(c,1))
#------------------------------------

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')


# In[ ]:


#Insertion_sort_1
import math
import os
import random
import re
import sys

#----------------------------------------
def insertionSort1(n, arr):
      x = arr[-1]

      for i in range(len(arr)-2, -1, -1):
          if arr[i] > x:
              arr[i+1] = arr[i]
              print(" ".join(map(str, arr)))
          else:
              arr[i+1] = x
              print(" ".join(map(str, arr)))
              break
      if arr[0] > x:
          arr[0] = x
          print(" ".join(map(str, arr)))
#-----------------------------------------

if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)


# In[ ]:


#Insertion_sort_2
import math
import os
import random
import re
import sys
#--------------------------------------
def insertionSort2(n, arr):
    for i in range(1,n):
        x = arr[i]
        j = i-1
        while j>=0 and x < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = x
        print(*arr)
#--------------------------------------
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

