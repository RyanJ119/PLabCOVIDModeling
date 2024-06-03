#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:09:22 2023

@author: ryanweightman
"""

from casadi import *
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import os
from datetime import datetime
import math

model="ModelV3"

def solve_control_problem(problem, max_num_vaccines_per_day, init_S=None,
                          init_E=None, init_I=None, init_R=None, init_V=None,
                          init_w=None, init_u=None):
    ## Parameters
    T = problem.time_horizon
    N = problem.N
    a = problem.contact_matrix
    gamma = problem.gamma
    delta = problem.delta
    cost_of_lockdown=  70
    cost_of_lockdown_old=   70
    cost_of_lockdown_school = 70
    cost_per_death= 1500000
    tau1 = 1 #4/5
    tau2 = 1 #2/3
    upper_bound = inf
    w_min = [0, 0, 0] 
    w_max=[1, 1, 1] 
    beta = problem.R0 * gamma
    death_rates = problem.death_rates
    initial_S = problem.initial_S
    initial_E = problem.initial_E
    initial_I = problem.initial_I
    initial_R = problem.initial_R
    numControls = 3
    tab_N= problem.population
    num_age_groups = tab_N.shape[1]

    Ntot=sum2(tab_N)  # total population in NJ
   

    ## Declaration of variables
    S=MX.sym('S', N+1, num_age_groups)
    E=MX.sym('E', N+1, num_age_groups)
    I=MX.sym('I', N+1, num_age_groups)
    R=MX.sym('R', N+1, num_age_groups)
   # V=MX.sym('V',N+1,num_age_groups)
    dSdt=MX.sym('dSdt', N+1, num_age_groups)
    dEdt=MX.sym('dEdt', N+1, num_age_groups)
    dIdt=MX.sym('dIdt', N+1, num_age_groups)
    dRdt=MX.sym('dRdt', N+1, num_age_groups)
   
    u=MX.sym('u',N+1)
    w=MX.sym('w',N+1, numControls)

################################### Building Matrices for age based control
    dimensions = np.shape(a)
    rows, columns = dimensions
    mat_old= a.copy()
    mat_only_old=a.copy()
    
    for i in range(rows):
        for j in range(columns):
            mat_old[i][j] = mat_old[i][j]*tau1
            mat_only_old[i][j] = mat_only_old[i][j]*tau1
            if i<rows-1:
                if j<columns-1:
                    mat_old[i][j]=0  #mat_old is the interactions that the elderly have with all other populations
            if i<rows-1 or j<columns-1:
                        mat_only_old[i][j] =0
                        
    mat_school=a.copy()
    
    for i in range(rows):
        for j in range(columns):
            mat_school[i][j] = mat_school[i][j]*tau2
            if (i != 0 and i!=1 and i!=2 and i!=3 ) or (j != 0 and j!=1 and j!=2 and j!=3):
                mat_school[i][j]=0             #mat_school is the interactions children have with each other
  
    matrix4 = a -    (mat_old +   mat_school  ) 
    #matrix4=a.copy()
#####################################

    
    ## Discretization of dynamics with implicit RK2
    dSdt = ( -1*( ((1-w[:,2]) *(1-w[:,0]) * beta * S * (mtimes(I,mat_old-mat_only_old)) ) +((1-w[:,0])*(1-w[:,0]) * beta * S * (mtimes(I,mat_only_old)) ) +((1-w[:,1]) *(1-w[:,1]) * beta * S * (mtimes(I,mat_school))) +(1-w[:,2]) *(1-w[:,2]) *(beta * S * (mtimes(I,matrix4))) )/ repmat(mtimes(tab_N, a), N+1, 1) )  #+sigma*R
        
    dEdt = ( ((1-w[:,2]) *(1-w[:,0]) * beta * S * (mtimes(I,mat_old-mat_only_old)) ) +((1-w[:,0])*(1-w[:,0]) * beta * S * (mtimes(I,mat_only_old)) ) +((1-w[:,1]) *(1-w[:,1]) * beta * S * (mtimes(I,mat_school))) +(1-w[:,2]) *(1-w[:,2]) *(beta * S * (mtimes(I,matrix4))) )/ repmat(mtimes(tab_N, a), N+1, 1) - delta * E
    
    dIdt = delta * E - gamma * I 
    
    dRdt = gamma * I   #- sigma*R


    
    cont_dynS = S[1:N+1,:] - S[0:N,:] - T/(2*N) * (dSdt[0:N,:] + dSdt[1:N+1,:])
    cont_dynE = E[1:N+1,:] - E[0:N,:] - T/(2*N) * (dEdt[0:N,:] + dEdt[1:N+1,:])
    cont_dynI = I[1:N+1,:] - I[0:N,:] - T/(2*N) * (dIdt[0:N,:] + dIdt[1:N+1,:])
    cont_dynR = R[1:N+1,:] - R[0:N,:] - T/(2*N) * (dRdt[0:N,:] + dRdt[1:N+1,:])




    cont_dyn = vertcat(
        reshape(cont_dynS,-1,1),
        reshape(cont_dynE,-1,1),
        reshape(cont_dynI,-1,1),
        reshape(cont_dynR,-1,1))
    
    
    ## Take into account the constraint  \sum_{j=1}^6 w_j(t) <= w_max
    gg = vertcat(cont_dyn, w[:,0], w[:,1], w[:,2])
   
    lower_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), np.concatenate((w_min[0]*np.ones((N+1)), w_min[1]*np.ones((N+1)), w_min[2]*np.ones((N+1))), axis=None))   # w_min=0
    upper_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), np.concatenate((w_max[0]*np.ones((N+1)), w_max[1]*np.ones((N+1)), w_max[2]*np.ones((N+1))), axis=None)) 

    cost_deaths = sum2(R[N, :] * death_rates)*cost_per_death
    cost_lockdown=sum2((sum1( mat_old) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_old*sum2(w[:,0])) + sum2((sum1( mat_school) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_school*sum2(w[:,1]))+ sum2((sum1( matrix4) / sum1(a)) *tab_N )*sum1(cost_of_lockdown*sum2(w[:,2]))
    cost_end = sum2(I[N, :] * death_rates)*cost_per_death*90
    cost_all=cost_deaths+cost_lockdown+cost_end

    ## Here we compute the dynamics for the control associated to a starting point
    print("Computation of the starting point:")

    if init_S is None:
        init_S=np.ones((N+1,1))* 2/3*initial_S
        init_E=np.ones((N+1,1))*2/3*initial_E
        init_I=np.ones((N+1,1))* 2/3*initial_I
        init_R=np.ones((N+1,1))* 2/3*initial_R
        
        init_w=pd.DataFrame.to_numpy(pd.read_csv('../Starting point/w.csv', delimiter=','))

    init_xu=vertcat(
        reshape(init_S,-1,1),
        reshape(init_E,-1,1),
        reshape(init_I,-1,1),
        reshape(init_R,-1,1),
        reshape(init_w,-1,1))

    lower_bound_S = np.zeros((N+1,num_age_groups))
    lower_bound_S[0,:] = initial_S
    upper_bound_S =  upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_S[0,:] = initial_S

    lower_bound_E = np.zeros((N+1,num_age_groups))
    lower_bound_E[0,:]= initial_E
    upper_bound_E = upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_E[0,:] = initial_E

    lower_bound_I = np.zeros((N+1,num_age_groups))
    lower_bound_I[0,:] = initial_I
    upper_bound_I = upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_I[0,:] = initial_I

    lower_bound_R =  np.zeros((N+1,num_age_groups))
    lower_bound_R[0,:] = initial_R
    upper_bound_R = upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_R[0,:] = initial_R


    lower_bound_w = init_w
    upper_bound_w = init_w
    
    lower_bound_xu = vertcat(
        reshape(lower_bound_S,-1,1),
        reshape(lower_bound_E,-1,1),
        reshape(lower_bound_I,-1,1),
        reshape(lower_bound_R,-1,1),
        reshape(lower_bound_w,-1,1))

    upper_bound_xu = vertcat(
        reshape(upper_bound_S,-1,1),
        reshape(upper_bound_E,-1,1),
        reshape(upper_bound_I,-1,1),
        reshape(upper_bound_R,-1,1),
        reshape(upper_bound_w,-1,1))

    ## Solve
    optim_problem = {
        'x' : vertcat(
            reshape(S, -1, 1),
            reshape(E, -1, 1),
            reshape(I, -1, 1),
            reshape(R, -1, 1),
            reshape(w, -1, 1)),
        'f' : cost_all, # You might change the cost here.
        'g' : gg}
    options = {'error_on_fail': False}
    options['ipopt.print_frequency_iter'] = 100
    options['print_time'] = 0
    options['ipopt.max_iter'] = 10000
    solver = nlpsol('solver', 'ipopt', optim_problem, options)
    result = solver(x0=init_xu,                   ## initialization
                    lbx=lower_bound_xu,           ## lower bounds on the global unknown x
                    ubx=upper_bound_xu,           ## upper bounds on the global unknown x
                    lbg=lower_bound_gg,           ## lower bounds on g
                    ubg=upper_bound_gg)           ## upper bounds on g

    cost=result['f']; xu=result['x']; g=result['g'] ## must be 0
    S=xu[:(1 * num_age_groups * (N+1))]
    S=reshape(S,N+1, num_age_groups)
    E=xu[(1 * num_age_groups * (N+1)):(2 * num_age_groups * (N+1))]
    E=reshape(E,N+1, num_age_groups)
    I=xu[(2 * num_age_groups * (N+1)):(3 * num_age_groups * (N+1))]
    I=reshape(I,N+1, num_age_groups)
    R=xu[(3 * num_age_groups * (N+1)):(4 * num_age_groups * (N+1))]
    R=reshape(R,N+1, num_age_groups)
    w=xu[(4 * num_age_groups * (N+1)):(4 * num_age_groups * (N+1))+(numControls*(N+1))]
    w=reshape(w,N+1,numControls)
    print(cost)


    init_S=np.array(S)
    init_E=np.array(E)
    init_I=np.array(I)
    init_R=np.array(R)
        
        
    init_w=pd.DataFrame.to_numpy(pd.read_csv('../Starting point/w.csv', delimiter=','))


# Now we look for the optimum

    ## Declaration of variables
    S=MX.sym('S', N+1, num_age_groups)
    E=MX.sym('E', N+1, num_age_groups)
    I=MX.sym('I', N+1, num_age_groups)
    R=MX.sym('R', N+1, num_age_groups)
   # V=MX.sym('V',N+1,num_age_groups)
    dSdt=MX.sym('dSdt', N+1, num_age_groups)
    dEdt=MX.sym('dEdt', N+1, num_age_groups)
    dIdt=MX.sym('dIdt', N+1, num_age_groups)
    dRdt=MX.sym('dRdt', N+1, num_age_groups)

    w=MX.sym('w',N+1, numControls)

    
    ## Discretization of dynamics with implicit RK2
    dSdt = ( -1*( ((1-w[:,2]) *(1-w[:,0]) * beta * S * (mtimes(I,mat_old-mat_only_old)) ) +((1-w[:,0])*(1-w[:,0]) * beta * S * (mtimes(I,mat_only_old)) ) +((1-w[:,1]) *(1-w[:,1]) * beta * S * (mtimes(I,mat_school))) +(1-w[:,2]) *(1-w[:,2]) *(beta * S * (mtimes(I,matrix4))) )/ repmat(mtimes(tab_N, a), N+1, 1) )  #+sigma*R
        
    dEdt = ( ((1-w[:,2]) *(1-w[:,0]) * beta * S * (mtimes(I,mat_old-mat_only_old)) ) +((1-w[:,0])*(1-w[:,0]) * beta * S * (mtimes(I,mat_only_old)) ) +((1-w[:,1]) *(1-w[:,1]) * beta * S * (mtimes(I,mat_school))) +(1-w[:,2]) *(1-w[:,2]) *(beta * S * (mtimes(I,matrix4))) )/ repmat(mtimes(tab_N, a), N+1, 1) - delta * E
    
    dIdt = delta * E - gamma * I 
    
    dRdt = gamma * I   #- sigma*R


    
    cont_dynS = S[1:N+1,:] - S[0:N,:] - T/(2*N) * (dSdt[0:N,:] + dSdt[1:N+1,:])
    cont_dynE = E[1:N+1,:] - E[0:N,:] - T/(2*N) * (dEdt[0:N,:] + dEdt[1:N+1,:])
    cont_dynI = I[1:N+1,:] - I[0:N,:] - T/(2*N) * (dIdt[0:N,:] + dIdt[1:N+1,:])
    cont_dynR = R[1:N+1,:] - R[0:N,:] - T/(2*N) * (dRdt[0:N,:] + dRdt[1:N+1,:])




    cont_dyn = vertcat(
        reshape(cont_dynS,-1,1),
        reshape(cont_dynE,-1,1),
        reshape(cont_dynI,-1,1),
        reshape(cont_dynR,-1,1))
    
    
    ## Take into account the constraint  \sum_{j=1}^6 w_j(t) <= w_max
    gg = vertcat(cont_dyn, w[:,0], w[:,1], w[:,2])
   
    lower_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), np.concatenate((w_min[0]*np.ones((N+1)), w_min[1]*np.ones((N+1)), w_min[2]*np.ones((N+1))), axis=None))   # w_min=0
    upper_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), np.concatenate((w_max[0]*np.ones((N+1)), w_max[1]*np.ones((N+1)), w_max[2]*np.ones((N+1))), axis=None)) 
    
    cost_deaths = sum2(R[N, :] * death_rates)*cost_per_death
    cost_lockdown=sum2((sum1( mat_old) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_old*sum2(w[:,0])) + sum2((sum1( mat_school) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_school*sum2(w[:,1]))+ sum2((sum1( matrix4) / sum1(a)) *tab_N )*sum1(cost_of_lockdown*sum2(w[:,2]))
    cost_end = sum2(I[N, :] * death_rates)*cost_per_death*90
    cost_all=cost_deaths+cost_lockdown+cost_end




    print("Computation of the optimum:")


    init_xu=vertcat(
        reshape(init_S,-1,1),
        reshape(init_E,-1,1),
        reshape(init_I,-1,1),
        reshape(init_R,-1,1),
        reshape(init_w,-1,1))
    print(init_w)

    lower_bound_S = np.zeros((N+1,num_age_groups))
    lower_bound_S[0,:] = initial_S
    upper_bound_S =  upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_S[0,:] = initial_S

    lower_bound_E = np.zeros((N+1,num_age_groups))
    lower_bound_E[0,:]= initial_E
    upper_bound_E = upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_E[0,:] = initial_E

    lower_bound_I = np.zeros((N+1,num_age_groups))
    lower_bound_I[0,:] = initial_I
    upper_bound_I = upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_I[0,:] = initial_I

    lower_bound_R =  np.zeros((N+1,num_age_groups))
    lower_bound_R[0,:] = initial_R
    upper_bound_R = upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_R[0,:] = initial_R

    lower_bound_w = w_min*np.ones((N+1,numControls))
    upper_bound_w = w_max*np.ones((N+1,numControls))

    lower_bound_xu = vertcat(
        reshape(lower_bound_S,-1,1),
        reshape(lower_bound_E,-1,1),
        reshape(lower_bound_I,-1,1),
        reshape(lower_bound_R,-1,1),
        reshape(lower_bound_w,-1,1))

    upper_bound_xu = vertcat(
        reshape(upper_bound_S,-1,1),
        reshape(upper_bound_E,-1,1),
        reshape(upper_bound_I,-1,1),
        reshape(upper_bound_R,-1,1),
        reshape(upper_bound_w,-1,1))

    ## Solve
    optim_problem = {
        'x' : vertcat(
            reshape(S, -1, 1),
            reshape(E, -1, 1),
            reshape(I, -1, 1),
            reshape(R, -1, 1),
            reshape(w, -1, 1)),
        'f' : cost_all, # You might change the cost here.
        'g' : gg}
    options = {'error_on_fail': False}
    options['ipopt.print_frequency_iter'] = 100
    options['print_time'] = 0
    options['ipopt.max_iter'] = 10000
    options['ipopt.slack_bound_frac'] = 0.5
    solver = nlpsol('solver', 'ipopt', optim_problem, options)
    result = solver(x0=init_xu,                   ## initialization
                    lbx=lower_bound_xu,           ## lower bounds on the global unknown x
                    ubx=upper_bound_xu,           ## upper bounds on the global unknown x
                    lbg=lower_bound_gg,           ## lower bounds on g
                    ubg=upper_bound_gg)           ## upper bounds on g

    cost=result['f']; xu=result['x']; g=result['g'] ## must be 0
    S=xu[:(1 * num_age_groups * (N+1))]
    S=reshape(S,N+1, num_age_groups)
    E=xu[(1 * num_age_groups * (N+1)):(2 * num_age_groups * (N+1))]
    E=reshape(E,N+1, num_age_groups)
    I=xu[(2 * num_age_groups * (N+1)):(3 * num_age_groups * (N+1))]
    I=reshape(I,N+1, num_age_groups)
    R=xu[(3 * num_age_groups * (N+1)):(4 * num_age_groups * (N+1))]
    R=reshape(R,N+1, num_age_groups)
    w=xu[(4 * num_age_groups * (N+1)):(4 * num_age_groups * (N+1))+(numControls*(N+1))]
    w=reshape(w,N+1,numControls)
    print(cost)
    return S, E, I, R, w,cost
