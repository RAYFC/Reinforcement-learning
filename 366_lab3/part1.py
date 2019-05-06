import numpy as np
import matplotlib.pyplot as plt
def v_iteration(V,R):
    global p,theta
    Pi=np.zeros(100)
    Pi[0]=None #we don't care the the policy of 0 and 100
    I=[] #information for all sweeps
    while True:
        delta=0
        temp_list1=[] #store the information for this step
        for state in range(1,100): # only consider state 1 .... 99 not include 0 and 100
            temp=V[state] #the estimate value 
            temp_list=[] #store all the returns of diferent actions
            for action in range(1, min(state, 100 - state)+1): #for all possible actions
                temp_list.append(p * (R[state + action] + V[state + action]) +
                (1 - p) * (R[state - action] + V[state - action])) 
            V[state]=max(temp_list) #find out the best action   
            Pi[state]=temp_list.index(V[state])+1 #update the policy,since we we start from state 1 in line 11, 0 represents stage 1 ,1 represents 2 ...............
            delta = max(delta, abs(temp - V[state])) #update the biggest diffenrence of all states now
            temp_list1.append(V[state])   #append the value of this state to the values of all states in this sweep
        I.append(temp_list1)                #append all values of states of this sweep to information
        if delta <= theta:      #when the biggest dif is small enough
            break
    return Pi,I
def main():
    global p,theta 
    p= 0.25
    theta=0.000001
    Value=np.zeros(101) #state 0-100 the index is exactly the # of the state
    Reward = np.zeros(101) #Rewards of state 0-100
    Reward[100] = 1  #all rewards are 0 except state 100
    Policy,Information = v_iteration(Value,Reward)
    Sweep = len(Information) #How many lists are there? = # of sweeps
    plt.figure(1)
    plt.suptitle('p = ' + str(p))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    for i in range(Sweep):
        plt.plot(Information[i]) #plot for every sweep
    plt.figure(2)
    plt.suptitle('p = ' + str(p))
    plt.xlabel('Capital')
    plt.ylabel('Final policy ')
    plt.plot(Policy)
    plt.show()

main()
