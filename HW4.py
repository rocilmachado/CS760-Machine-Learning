import math
import numpy as np
from pip._vendor.html5lib._ihatexml import letter

training_e = {}
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', " "]
for i in range(len(letters)):
    training_e[letters[i]] = 0
c0 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\e0.txt", "r").read()
c0 = c0.replace("\n", "")
for i in c0:
    if i in letters:
        training_e[i] = training_e[i] + 1
c1 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\e1.txt", "r").read()
c1 = c1.replace("\n", "")
for i in c1:
    if i in letters:
        training_e[i] = training_e[i] + 1
c2 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\e2.txt", "r").read()
c2 = c2.replace("\n", "")
for i in c2:
    if i in letters:
        training_e[i] = training_e[i] + 1
c3 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\e3.txt", "r").read()
c3 = c3.replace("\n", "")
for i in c3:
    if i in letters:
        training_e[i] = training_e[i] + 1
c4 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\e4.txt", "r").read()
c4 = c4.replace("\n", "")
for i in c4:
    if i in letters:
        training_e[i] = training_e[i] + 1
c5 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\e5.txt", "r").read()
c5 = c5.replace("\n", "")
for i in c5:
    if i in letters:
        training_e[i] = training_e[i] + 1
c6 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\e6.txt", "r").read()
c6 = c6.replace("\n", "")
for i in c6:
    if i in letters:
        training_e[i] = training_e[i] + 1
c7 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\e7.txt", "r").read()
c7 = c7.replace("\n", "")
for i in c7:
    if i in letters:
        training_e[i] = training_e[i] + 1
c8 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\e8.txt", "r").read()
c8 = c8.replace("\n", "")
for i in c8:
    if i in letters:
        training_e[i] = training_e[i] + 1
c9 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\e9.txt", "r").read()
c9 = c9.replace("\n", "")
for i in c9:
    if i in letters:
        training_e[i] = training_e[i] + 1
print("Character count in training set e0-e9:")
for i in letters:
    print(i, training_e[i])
te = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
count_e = len(te)
print("Total variables in e0-e9:", count_e)

training_j = {}
#letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', " "]
for i in range(len(letters)):
    training_j[letters[i]] = 0
c0 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\j0.txt", "r").read()
c0 = c0.replace("\n", "")
for i in c0:
    if i in letters:
        training_j[i] = training_j[i] + 1
c1 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\j1.txt", "r").read()
c1 = c1.replace("\n", "")
for i in c1:
    if i in letters:
        training_j[i] = training_j[i] + 1
c2 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\j2.txt", "r").read()
c2 = c2.replace("\n", "")
for i in c2:
    if i in letters:
        training_j[i] = training_j[i] + 1
c3 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\j3.txt", "r").read()
c3 = c3.replace("\n", "")
for i in c3:
    if i in letters:
        training_j[i] = training_j[i] + 1
c4 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\j4.txt", "r").read()
c4 = c4.replace("\n", "")
for i in c4:
    if i in letters:
        training_j[i] = training_j[i] + 1
c5 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\j5.txt", "r").read()
c5 = c5.replace("\n", "")
for i in c5:
    if i in letters:
        training_j[i] = training_j[i] + 1
c6 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\j6.txt", "r").read()
c6 = c6.replace("\n", "")
for i in c6:
    if i in letters:
        training_j[i] = training_j[i] + 1
c7 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\j7.txt", "r").read()
c7 = c7.replace("\n", "")
for i in c7:
    if i in letters:
        training_j[i] = training_j[i] + 1
c8 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\j8.txt", "r").read()
c8 = c8.replace("\n", "")
for i in c8:
    if i in letters:
        training_j[i] = training_j[i] + 1
c9 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\j9.txt", "r").read()
c9 = c9.replace("\n", "")
for i in c9:
    if i in letters:
        training_j[i] = training_j[i] + 1
print("Character count in training set j0-j9:")
for i in letters:
    print(i, training_j[i])
tj = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
count_j = len(tj)
print("Total variables in j0-j9:", count_j)


training_s= {}
#letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', " "]
for i in range(len(letters)):
    training_s[letters[i]] = 0
c0 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\s0.txt", "r").read()
c0 = c0.replace("\n", "")
for i in c0:
    if i in letters:
        training_s[i] = training_s[i] + 1
c1 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\s1.txt", "r").read()
c1 = c1.replace("\n", "")
for i in c1:
    if i in letters:
        training_s[i] = training_s[i] + 1
c2 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\s2.txt", "r").read()
c2 = c2.replace("\n", "")
for i in c2:
    if i in letters:
        training_s[i] = training_s[i] + 1
c3 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\s3.txt", "r").read()
c3 = c3.replace("\n", "")
for i in c3:
    if i in letters:
        training_s[i] = training_s[i] + 1
c4 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\s4.txt", "r").read()
c4 = c4.replace("\n", "")
for i in c4:
    if i in letters:
        training_s[i] = training_s[i] + 1
c5 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\s5.txt", "r").read()
c5 = c5.replace("\n", "")
for i in c5:
    if i in letters:
        training_s[i] = training_s[i] + 1
c6 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\s6.txt", "r").read()
c6 = c6.replace("\n", "")
for i in c6:
    if i in letters:
        training_s[i] = training_s[i] + 1
c7 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\s7.txt", "r").read()
c7 = c7.replace("\n", "")
for i in c7:
    if i in letters:
        training_s[i] = training_s[i] + 1
c8 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\s8.txt", "r").read()
c8 = c8.replace("\n", "")
for i in c8:
    if i in letters:
        training_s[i] = training_s[i] + 1
c9 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\s9.txt", "r").read()
c9 = c9.replace("\n", "")
for i in c9:
    if i in letters:
        training_s[i] = training_s[i] + 1
print("Character count in training set s0-s9:")
for i in letters:
    print(i, training_s[i])
ts = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
count_s = len(ts)
print("Total variables in s0-s9:", count_s)

########################Find conditional probabilities of a,b,c,....space given y = e, y = j, y = s################
cond_prob_y_e = {}
cond_prob_y_j = {}
cond_prob_y_s = {}
#letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
for i in range (len(letters)):
    cond_prob_y_e[letters[i]] = (training_e[letters[i]])/(count_e)
    cond_prob_y_j[letters[i]] = (training_j[letters[i]])/(count_j)
    cond_prob_y_s[letters[i]] = (training_s[letters[i]])/(count_s)
print("Conditional probabilities before smoothing:")
#print("P(x=char| y= e):",cond_prob_y_e, ", P(x=char| y= j):", cond_prob_y_j, ", P(x=char| y= s):", cond_prob_y_s)
print("Conditional probabilities P(X = c|Y=e)before smoothing:")
for i in letters:
    #print("P(x=",i,"| y= e):", cond_prob_y_e[i])
    print(cond_prob_y_e[i])
print("Conditional probabilities P(X = c|Y=j)before smoothing:")
for i in letters:
    #print("P(x=",i,"| y= j):", cond_prob_y_j[i])
    print(cond_prob_y_j[i])
print("Conditional probabilities P(X = c|Y=s)before smoothing:")
for i in letters:
    #print("P(x=",i,"| y= s):", cond_prob_y_s[i])
    print(cond_prob_y_s[i])

################################### Add-1 Smoothing####################################################################
for i in range (len(letters)):
    cond_prob_y_e[letters[i]] = (training_e[letters[i]] + 1)/(count_e + 27)
    cond_prob_y_j[letters[i]] = (training_j[letters[i]] + 1)/(count_j + 27)
    cond_prob_y_s[letters[i]] = (training_s[letters[i]] + 1)/(count_s + 27)
print("Conditional probabilities after Add-1 smoothing:")
print("Conditional probabilities P(X=c|Y=e) after Add-1 smoothing:")
for i in letters:
    #print("P(x=",i,"| y= e):", cond_prob_y_e[i])
    print(cond_prob_y_e[i])
print("Conditional probabilities P(X=c|Y=j) after Add-1 smoothing:")
for i in letters:
    #print("P(x=",i,"| y= j):", cond_prob_y_j[i])
    print(cond_prob_y_j[i])
print("Conditional probabilities P(X=c|Y=s) after Add-1 smoothing:")
for i in letters:
    #print("P(x=",i,"| y= s):", cond_prob_y_s[i])
    print(cond_prob_y_s[i])

#################Print character counts in langauage file e10.txt################################################
char_count_e = {}
for i in range(len(letters)):
    char_count_e[letters[i]] = 0
e10 = open(r"C:\Users\Rocil\Documents\Machine Learning\HW4\languageID\e10.txt", "r").read()
e10 = e10.replace("\n", "")
for i in e10:
    if i in letters:
        char_count_e[i] = char_count_e[i] + 1
#print("Character count in e10:", char_count_e)

####################Q5 calculate P(X|Y=e), P(X|Y=j), P(X|Y=s) for e10.txt#############################################################
p_x_given_y_e = 0
p_x_given_y_j = 0
p_x_given_y_s = 0
for i in letters:
    p_x_given_y_e = p_x_given_y_e + char_count_e[i] * math.log10(cond_prob_y_e[i]) 
    p_x_given_y_j = p_x_given_y_j + char_count_e[i] * math.log10(cond_prob_y_j[i]) 
    p_x_given_y_s = p_x_given_y_s + char_count_e[i] * math.log10(cond_prob_y_s[i])

print("Log of P(X|Y=e):", p_x_given_y_e, "P(X|Y=j):", p_x_given_y_j, "P(X|Y=s):", p_x_given_y_s)

####################Q6 calculate P(Y=e|X), P(Y=j|X), P(Y=s|X) for e10.txt#############################################################

log_p_y_e_given_x = 0
log_p_y_j_given_x = 0
log_p_y_s_given_x = 0

log_p_y_e_given_x = p_x_given_y_e + math.log10(11/33)
log_p_y_j_given_x = p_x_given_y_j + math.log10(11/33)
log_p_y_s_given_x = p_x_given_y_s + math.log10(11/33)

print("Log of P(Y=e|X), P(Y=j|X), P(Y=s|X):", log_p_y_e_given_x, log_p_y_j_given_x, log_p_y_s_given_x)

p_y_e_given_x = pow(10, log_p_y_e_given_x)
p_y_j_given_x = pow(10, log_p_y_j_given_x)
p_y_s_given_x = pow(10, log_p_y_s_given_x)

max_posterior = max(log_p_y_e_given_x,log_p_y_j_given_x,log_p_y_s_given_x)
print("The max posterior probability of test file is :", max_posterior)
if(max_posterior == log_p_y_e_given_x):
    label = "English"
elif(max_posterior == log_p_y_j_given_x):
    label  = "Japanese"
else:
    label = "Spanish"
print("The predicted label is:", label)

#print("Taking antilog P(Y=e|X), P(Y=j|X), P(Y=s|X):", p_y_e_given_x, p_y_j_given_x, p_y_s_given_x)



####Using Bayes Theorem###################################################################
#  P(y|x) = P(X|Y)P(Y)/P(x)
##########################################################################################


    
    