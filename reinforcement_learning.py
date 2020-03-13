'''
Created on Nov 26, 2019

@author: Rocil
'''

import numpy as np
import random
from random import *
from numpy import empty
items = [0, 1]

# Initialize q-table values to 0
state_size=2
action_size=2
Q = np.zeros((state_size, action_size))
state=0
#action=0

#move=0 and stay=1
#A=0 and B=1
'''

#PART 1
for i in range(0,200):
    action= sample(items, 1) 
    print('Random action selected:', action)
    if action[0] == 0:
        if state==0:
            newstate=1
        else:
            newstate=0
        Q[state, action[0]] = 0.5 * Q[state, action[0]] + 0.5 * (action[0] + 0.9 * np.max(Q[newstate, :]))
        state=newstate
            
    else:
        Q[state, action[0]] = 0.5 * Q[state, action[0]] + 0.5 * (action[0] + 0.9 * np.max(Q[state, :]))
            
    print(Q)
    print('My new state is:', state)



action=[0]
#PART 2
for i in range(0,200):
    choice= sample(items,  1)   # Pick a random item from the list  
    print('Choice is', choice)
    if choice[0]== 0:
        print('Current state:', state, "Current action", action)
        if state == 0 and action[0]== 0:
            if Q[1,0] > Q[0,1] or Q[1,0] == Q[0,1]:
                newaction=0
                newstate=1
            else:
                newaction=1
                newstate=0
        elif state==0 and action[0]==1:
            if Q[1,0] > Q[0,1] or Q[1,0] == Q[0,1]:
                newaction=0
                newstate=1
            else:
                newaction=1
                newstate=0
        elif state==1 and action[0]==0:
            if Q[0,0]> Q[1,1] or Q[0,0] == Q[1,1]:
                newstate=0
                newaction=0
            else:
                newstate=1
                newaction=1
        elif state==1 and action[0]==1:
            if Q[0,0]>Q[1,1] or Q[0,0] == Q[1,1]:
                newstate=0
                newaction=0
            else:
                newstate=1
                newaction=1
           
        if newaction == 0:
            Q[state, newaction] = 0.5 * Q[state, newaction] + 0.5 * (newaction + 0.9 * np.max(Q[newstate, :]))
        else:
            Q[state, newaction] = 0.5 * Q[state, newaction] + 0.5 * (newaction + 0.9 * np.max(Q[state, :]))
        print(Q)
        state=newstate
        print('My new state is:', state)
        action[0]=newaction
        print("Therefore, action chosen from Q table is:", action)
    
    elif choice[0]==1:
        action= sample(items, 1) 
        print('Random action selected:', action)
        if action[0] == 0:
            if state==0:
                newstate=1
            else:
                newstate=0
            Q[state, action[0]] = 0.5 * Q[state, action] + 0.5 * (action[0] + 0.9 * np.max(Q[newstate, :]))
            state=newstate
            
        else:
            Q[state, action[0]] = 0.5 * Q[state, action] + 0.5 * (action[0] + 0.9 * np.max(Q[state, :]))
            
        print(Q)
        print('My new state is:', state)
'''
#PART 3
action=0
for i in range(0,200):
    if state == 0 and action == 0: #Current state A
        if Q[1,0] > Q[0,1] or Q[1,0] == Q[0,1]:
            newaction=0  #move to
            newstate=1   #State B
        else:
            newaction=1
            newstate=0
    elif state==0 and action==1:
        if Q[1,0] > Q[0,1] or Q[1,0] == Q[0,1]:
            newaction=0
            newstate=1
        else:
            newaction=1
            newstate=0
    elif state==1 and action==0:
        if Q[0,0]> Q[1,1] or Q[0,0] == Q[1,1]:
            newstate=0
            newaction=0
        else:
            newstate=1
            newaction=1
    elif state==1 and action==1:
        if Q[0,0]>Q[1,1] or Q[0,0] == Q[1,1]:
            newstate=0
            newaction=0
        else:
            newstate=1
            newaction=1
           
    if newaction == 0:
        Q[state, newaction] = 0.5 * Q[state, newaction] + 0.5 * (newaction + 0.9 * np.max(Q[newstate, :]))
    else:
        Q[state, newaction] = 0.5 * Q[state, newaction] + 0.5 * (newaction + 0.9 * np.max(Q[state, :]))
    print(Q)
    state=newstate
    action=newaction
