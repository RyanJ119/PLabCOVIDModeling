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

model="Adding_Transports"

commuting_proportions=np.array([
    0.0,
    0.0,
    0.0636524635249281,
    0.0636524635249281,
    0.13547943082912334,
    0.13547943082912334,
    0.11524207023278385,
    0.11524207023278385,
    0.07578589407205107,
    0.07578589407205107,
    0.07578589407205107
])

omega=0.2 # This is the proportion of interactions a Commute users has in the Public Transports

def solve_control_problem(problem, max_num_vaccines_per_day, init_S=None,
                          init_E=None, init_I=None, init_R=None, init_V=None,
                          init_w=None, init_u=None):
    ## Parameters
    T = problem.time_horizon
    N = problem.N
    a = problem.contact_matrix
    gamma = problem.gamma
   # sigma = 1/60
    delta = problem.delta
    cost_of_lockdown=  70
    cost_of_lockdown_old=   70
    cost_of_lockdown_school = 70
    cost_per_death= 1500000
    tau1 = 1 #4/5
    tau2 = 1 #2/3
    upper_bound = inf
    w_min = [0, 0, 0, 0]
    w_max=[1, 1, 1, 1]  
    beta = problem.R0 * gamma
    death_rates = problem.death_rates
    initial_S = problem.initial_S
    initial_E = problem.initial_E
    initial_I = problem.initial_I
    initial_R = problem.initial_R
    numControls = 4
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
    
    mat_old_transport=omega*commuting_proportions*mat_old.copy()
    mat_old-=mat_old_transport
    
    mat_only_old_transport=omega*commuting_proportions*mat_only_old.copy()
    mat_only_old-=mat_only_old_transport
    
    mat_school_transport=omega*commuting_proportions*mat_school.copy()
    mat_school-=mat_school_transport
    
    matrix4_transport=omega*commuting_proportions*matrix4.copy()
    matrix4-=matrix4_transport
#####################################

    
    ## Discretization of dynamics with implicit RK2
    dSdt = ( -1*( 
         # The following block corresponds to the interactions that happen when commuting

         (1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * S * mtimes(I,mat_old_transport-mat_only_old_transport)
            +(1-w[:,0])*(1-w[:,0]) * beta * S * mtimes(I,mat_only_old_transport) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * S * (mtimes(I,mat_school_transport)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * S * mtimes(I,matrix4_transport) 
         )
        # The following blocks correspond to the interactions that are indirectly impacted by the closure of public transports in a linear way
        +(1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*commuting_proportions).T * (mtimes((I.T*(1-commuting_proportions)).T,mat_school))
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,matrix4) 

            +(1-w[:,2]) *(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*(1-commuting_proportions)).T * (mtimes((I.T*commuting_proportions).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,matrix4) 
         )
         # The following blocks correspond to the interactions that are indirectly impacted by the closure of public transports in a quadratic way
        +(1-w[:,3])*(1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*commuting_proportions).T * (mtimes((I.T*commuting_proportions).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,matrix4) 
         )
         # The following blocks correspond to the interactions that are not impacted by the closure of public transports
         +(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*(1-commuting_proportions)).T * (mtimes((I.T*(1-commuting_proportions)).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,matrix4) 
         )


         )/ repmat(mtimes(tab_N, a), N+1, 1) )
    
    dEdt = ( 
         # The following block corresponds to the interactions that happen when commuting

         (1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * S * mtimes(I,mat_old_transport-mat_only_old_transport)
            +(1-w[:,0])*(1-w[:,0]) * beta * S * mtimes(I,mat_only_old_transport) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * S * (mtimes(I,mat_school_transport)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * S * mtimes(I,matrix4_transport) 
         )
        # The following blocks correspond to the interactions that are indirectly impacted by the closure of public transports in a linear way
        +(1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*commuting_proportions).T * (mtimes((I.T*(1-commuting_proportions)).T,mat_school))
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,matrix4) 

            +(1-w[:,2]) *(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*(1-commuting_proportions)).T * (mtimes((I.T*commuting_proportions).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,matrix4) 
         )
         # The following blocks correspond to the interactions that are indirectly impacted by the closure of public transports in a quadratic way
        +(1-w[:,3])*(1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*commuting_proportions).T * (mtimes((I.T*commuting_proportions).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,matrix4) 
         )
         # The following blocks correspond to the interactions that are not impacted by the closure of public transports
         +(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*(1-commuting_proportions)).T * (mtimes((I.T*(1-commuting_proportions)).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,matrix4) 
         )


         )/ repmat(mtimes(tab_N, a), N+1, 1) - delta * E
    
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
    gg = vertcat(cont_dyn, w[:,0], w[:,1], w[:,2],w[:,3])
    
    lower_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), np.concatenate((w_min[0]*np.ones((N+1)), w_min[1]*np.ones((N+1)), w_min[2]*np.ones((N+1)), w_min[3]*np.ones((N+1))), axis=None))   # w_min=0
    upper_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), np.concatenate((w_max[0]*np.ones((N+1)), w_max[1]*np.ones((N+1)), w_max[2]*np.ones((N+1)), w_max[3]*np.ones((N+1))), axis=None)) 

    cost_deaths = sum2(R[N, :] * death_rates)*cost_per_death 
    cost_lockdown=(
        sum2((sum1(commuting_proportions * mat_old) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_old*sum2(1-(1-w[:,0])*(1-w[:,3]))) 
        + sum2((sum1(commuting_proportions * mat_school) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_school*sum2(1-(1-w[:,1])*(1-w[:,3])))
        + sum2((sum1(commuting_proportions * matrix4) / sum1(a)) *tab_N )*sum1(cost_of_lockdown*sum2(1-(1-w[:,2])*(1-w[:,3])))
        
        + sum2((sum1((1-commuting_proportions) * mat_old) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_old*sum2(w[:,0])) 
        + sum2((sum1((1-commuting_proportions) * mat_school) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_school*sum2(w[:,1]))
        + sum2((sum1((1-commuting_proportions) * matrix4) / sum1(a)) *tab_N )*sum1(cost_of_lockdown*sum2(w[:,2]))
        )
    cost_end = sum2(I[N, :] * death_rates)*cost_per_death*90
    cost_all=cost_deaths+cost_lockdown+cost_end


    ## Here we compute the dynamics for the control associated to a starting point

    print("Computation of the starting point:")

    if init_S is None:
        init_S=np.ones((N+1,1))* 2/3*initial_S
        init_E=np.ones((N+1,1))*2/3*initial_E
        init_I=np.ones((N+1,1))* 2/3*initial_I
        init_R=np.ones((N+1,1))* 2/3*initial_R

        init_w=0.5*np.ones((N+1,numControls),dtype=float)
        init_w[:,:3]=pd.DataFrame.to_numpy(pd.read_csv('../Starting point/w.csv', delimiter=','))

    init_xu=vertcat(
        reshape(init_S,-1,1),
        reshape(init_E,-1,1),
        reshape(init_I,-1,1),
        reshape(init_R,-1,1),
        reshape(init_w,-1,1),
        init_u.copy())

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
        
        
    init_w=0.*np.ones((N+1,4),dtype=float)
    init_w[:,:3]=pd.DataFrame.to_numpy(pd.read_csv('../Starting point/w.csv', delimiter=','))






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
   
    u=MX.sym('u',N+1)
    w=MX.sym('w',N+1, numControls)

    
    ## Discretization of dynamics with implicit RK2
    dSdt = ( -1*( 
         # The following block corresponds to the interactions that happen when commuting

         (1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * S * mtimes(I,mat_old_transport-mat_only_old_transport)
            +(1-w[:,0])*(1-w[:,0]) * beta * S * mtimes(I,mat_only_old_transport) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * S * (mtimes(I,mat_school_transport)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * S * mtimes(I,matrix4_transport) 
         )
        # The following blocks correspond to the interactions that are indirectly impacted by the closure of public transports in a linear way
        +(1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*commuting_proportions).T * (mtimes((I.T*(1-commuting_proportions)).T,mat_school))
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,matrix4) 

            +(1-w[:,2]) *(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*(1-commuting_proportions)).T * (mtimes((I.T*commuting_proportions).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,matrix4) 
         )
         # The following blocks correspond to the interactions that are indirectly impacted by the closure of public transports in a quadratic way
        +(1-w[:,3])*(1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*commuting_proportions).T * (mtimes((I.T*commuting_proportions).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,matrix4) 
         )
         # The following blocks correspond to the interactions that are not impacted by the closure of public transports
         +(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*(1-commuting_proportions)).T * (mtimes((I.T*(1-commuting_proportions)).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,matrix4) 
         )


         )/ repmat(mtimes(tab_N, a), N+1, 1) )
    
    dEdt = ( 
         # The following block corresponds to the interactions that happen when commuting

         (1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * S * mtimes(I,mat_old_transport-mat_only_old_transport)
            +(1-w[:,0])*(1-w[:,0]) * beta * S * mtimes(I,mat_only_old_transport) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * S * (mtimes(I,mat_school_transport)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * S * mtimes(I,matrix4_transport) 
         )
        # The following blocks correspond to the interactions that are indirectly impacted by the closure of public transports in a linear way
        +(1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*commuting_proportions).T * (mtimes((I.T*(1-commuting_proportions)).T,mat_school))
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*commuting_proportions).T * mtimes((I.T*(1-commuting_proportions)).T,matrix4) 

            +(1-w[:,2]) *(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*(1-commuting_proportions)).T * (mtimes((I.T*commuting_proportions).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*commuting_proportions).T,matrix4) 
         )
         # The following blocks correspond to the interactions that are indirectly impacted by the closure of public transports in a quadratic way
        +(1-w[:,3])*(1-w[:,3])*(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*commuting_proportions).T * (mtimes((I.T*commuting_proportions).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*commuting_proportions).T * mtimes((I.T*commuting_proportions).T,matrix4) 
         )
         # The following blocks correspond to the interactions that are not impacted by the closure of public transports
         +(
            (1-w[:,2]) *(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,mat_old-mat_only_old)
            +(1-w[:,0])*(1-w[:,0]) * beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,mat_only_old) 
            +(1-w[:,1]) *(1-w[:,1]) * beta * (S.T*(1-commuting_proportions)).T * (mtimes((I.T*(1-commuting_proportions)).T,mat_school)) 
            +(1-w[:,2]) *(1-w[:,2]) *beta * (S.T*(1-commuting_proportions)).T * mtimes((I.T*(1-commuting_proportions)).T,matrix4) 
         )


         )/ repmat(mtimes(tab_N, a), N+1, 1) - delta * E
    
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
    gg = vertcat(cont_dyn, w[:,0], w[:,1], w[:,2],w[:,3])
   # lower_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), w_min * np.ones(3*(N+1)))   # w_min=0
   # upper_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), w_max * np.ones(3*(N+1)))
    
    lower_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), np.concatenate((w_min[0]*np.ones((N+1)), w_min[1]*np.ones((N+1)), w_min[2]*np.ones((N+1)), w_min[3]*np.ones((N+1))), axis=None))   # w_min=0
    upper_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), np.concatenate((w_max[0]*np.ones((N+1)), w_max[1]*np.ones((N+1)), w_max[2]*np.ones((N+1)), w_max[3]*np.ones((N+1))), axis=None)) 

    cost_deaths = sum2(R[N, :] * death_rates)*cost_per_death     #   cost_deaths = sum1(mtimes(I,death_rates.T))

    cost_lockdown=(
        sum2((sum1(commuting_proportions * mat_old) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_old*sum2(1-(1-w[:,0])*(1-w[:,3]))) 
        + sum2((sum1(commuting_proportions * mat_school) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_school*sum2(1-(1-w[:,1])*(1-w[:,3])))
        + sum2((sum1(commuting_proportions * matrix4) / sum1(a)) *tab_N )*sum1(cost_of_lockdown*sum2(1-(1-w[:,2])*(1-w[:,3])))
        
        + sum2((sum1((1-commuting_proportions) * mat_old) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_old*sum2(w[:,0])) 
        + sum2((sum1((1-commuting_proportions) * mat_school) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_school*sum2(w[:,1]))
        + sum2((sum1((1-commuting_proportions) * matrix4) / sum1(a)) *tab_N )*sum1(cost_of_lockdown*sum2(w[:,2]))
        )
    #cost_lockdown=sum2((sum1( mat_school) / sum1(a)) *tab_N )*sum1(cost_of_lockdown_school*sum2(w[:,1]))
    cost_end = sum2(I[N, :] * death_rates)*cost_per_death*90
    cost_smoothness=sum1(sum2(w[:N,3]*w[:N,3]+w[1:,3]*w[1:,3]-2*w[:N,3]*w[1:,3]))/N*10**7
    cost_all=cost_deaths+cost_lockdown+cost_end #+cost_smoothness




    print("Computation of the optimum:")


    init_xu=vertcat(
        reshape(init_S,-1,1),
        reshape(init_E,-1,1),
        reshape(init_I,-1,1),
        reshape(init_R,-1,1),
        reshape(init_w,-1,1),
        init_u.copy())
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



    lower_bound_u = u_min*np.ones(N+1)
    upper_bound_u = u_max*np.ones(N+1)
    lower_bound_w = w_min*np.ones((N+1,numControls))
    upper_bound_w = w_max*np.ones((N+1,numControls))
    
    #upper_bound_w[0:8, :] = 0   ## [Manu] this constraint means that we switch off the two first controls, right?
    #print(upper_bound_w)
    #print(upper_bound_w)
    lower_bound_xu = vertcat(
        reshape(lower_bound_S,-1,1),
        reshape(lower_bound_E,-1,1),
        reshape(lower_bound_I,-1,1),
        reshape(lower_bound_R,-1,1),
        reshape(lower_bound_w,-1,1),
        lower_bound_u)

    upper_bound_xu = vertcat(
        reshape(upper_bound_S,-1,1),
        reshape(upper_bound_E,-1,1),
        reshape(upper_bound_I,-1,1),
        reshape(upper_bound_R,-1,1),
        reshape(upper_bound_w,-1,1),
        upper_bound_u)

    ## Solve
    optim_problem = {
        'x' : vertcat(
            reshape(S, -1, 1),
            reshape(E, -1, 1),
            reshape(I, -1, 1),
            reshape(R, -1, 1),
            reshape(w, -1, 1),
            u,),
        'f' : cost_all, # You might change the cost here.
        'g' : gg}
    options = {'error_on_fail': False}
#     options['ipopt.linear_solver'] = 'mumps'  # ma27
#     options['ipopt.hessian_approximation'] = 'exact'   #'exact', 'limited-memory'
#     options['ipopt.tol'] = 1e-4
#     options['ipopt.acceptable_tol'] = 1e-4
#     options['ipopt.acceptable_constr_viol_tol'] = 1e-4
    # # options['ipopt.print_level'] = 0
    options['ipopt.print_frequency_iter'] = 100
    options['print_time'] = 0
    #options['ipopt.warm_start_init_point'] = 'yes'
    options['ipopt.max_iter'] = 10000
    # options['ipopt.expect_infeasible_problem'] = "no"
    options['ipopt.slack_bound_frac'] = 0.5
    # options['ipopt.start_with_resto'] = "no"
    # options['ipopt.required_infeasibility_reduction'] = 0.85
    # options['ipopt.acceptable_iter'] = 8
    solver = casadi.nlpsol('solver', 'ipopt', optim_problem, options)
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
   # V=xu[(4 * num_age_groups * (N+1)):(5 * num_age_groups * (N+1))]
   # V=reshape(V,N+1, num_age_groups)
    w=xu[(4 * num_age_groups * (N+1)):(4 * num_age_groups * (N+1))+(numControls*(N+1))]
    w=reshape(w,N+1,numControls)
    u=xu[((4 * num_age_groups * (N+1))+numControls * (N+1)):]
    print(cost,cost-sum1(sum2(w[:N,3]*w[:N,3]+w[1:,3]*w[1:,3]-2*w[:N,3]*w[1:,3]))/N*10**7)
    return S, E, I, R, w,cost
