from casadi import *
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import os
from datetime import datetime
import math


def solve_control_problem(problem, max_num_vaccines_per_day, init_S=None,
                          init_E=None, init_I=None, init_R=None, init_w=None, init_u=None):
    ## Parameters
    T = problem.time_horizon
    N = problem.N
    a = problem.contact_matrix
    gamma = problem.gamma
    delta = problem.delta
    cost_of_lockdown=problem.cost_lockdown
    cost_per_death=1500000
    u_min= 0
    u_max= 1  # bounds on u: if u_min=u_max then no lockdown
    beta = problem.R0 * gamma
    upper_bound = inf
    #w_min=0
    #w_max=max_num_vaccines_per_day  ## w_max is the upper threshold on w(t) = \sum_{j=1}^6 w_j(t)  (maximal number of vaccines per day)
    death_rates = problem.death_rates
    initial_S = problem.initial_S
    initial_E = problem.initial_E
    initial_I = problem.initial_I
    initial_R = problem.initial_R
    
    
################################### Building Matrices for age based control
    dimensions = np.shape(a)
    rows, columns = dimensions
    matrix2= a.copy()
    
    for i in range(rows):
        for j in range(columns):
            if i<rows-1:
                if j<columns-1:
                    matrix2[i][j]=0         #Matrix2 is the interactions that the elderly have with all other populations
                 
    matrix3=a.copy()
    
                
    for i in range(rows):
        for j in range(columns):
            if (i != 0 and i!=1 and i!=2 ) or (j != 0 and j!=1 and j!=2):
                matrix3[i][j]=0             #Matrix3 is the interactions children have with each other
  
    matrix4 = a -    matrix2 -   matrix3   
#####################################



    tab_N= problem.population
    num_age_groups = tab_N.shape[1]

    Ntot=sum2(tab_N)  # total population
    #initial_V =  np.zeros((1,num_age_groups))


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
    #dVdt=MX.sym('dVdt', N+1, num_age_groups)

    u=MX.sym('u',N+1)
    
    #w=MX.sym('w',N+1,num_age_groups)
    #print(repmat(mtimes(tab_N, matrix2), N+1, 1))
   
    ## Discretization of dynamics with implicit RK2 (S+E+I)
   
    dSdt =  (-u * beta * S * (mtimes(I,matrix2))  -u * beta * S * (mtimes(I,matrix3)) -u * beta * S * (mtimes(I,matrix4)) )/ repmat(mtimes(tab_N, a), N+1, 1)
   
    dSdt = -u * beta * S * ((mtimes(I,a)) / repmat(mtimes(tab_N, a), N+1, 1))  #*(S>=0)*(E>=0)*(I>=0)
    #dSdt[np.isnan(dSdt)] = 0
    dEdt =  u*beta * S * ((mtimes(I,a)) / repmat(mtimes(tab_N, a), N+1, 1))  - delta * E
    
    dIdt = delta * E - gamma * I 
    
    dRdt = gamma * I
    
   # dVdt = w #*(S>=0)*(E>=0)*(I>=0)
   # dSvdt = -u * Sv * ((.3 * mtimes(I,a)+ .09 * mtimes(Iv,a)) / repmat(mtimes(tab_N, a), N+1, 1)) + w * S/(S+E+I) #*(S>=0)*(E>=0)*(I>=0)
    
   # dEvdt =  u * Sv * ((.3 * mtimes(I,a)+ .09 * mtimes(Iv,a)) / repmat(mtimes(tab_N, a), N+1, 1))  - delta * Ev + w * E/(S+E+I)
    
    #dIvdt = delta * Ev  + w * I/(S+E+I)
    

   
    

    
     
    cont_dynS = S[1:N+1,:] - S[0:N,:] - T/(2*N) * (dSdt[0:N,:] + dSdt[1:N+1,:])
    cont_dynE = E[1:N+1,:] - E[0:N,:] - T/(2*N) * (dEdt[0:N,:] + dEdt[1:N+1,:])
    cont_dynI = I[1:N+1,:] - I[0:N,:] - T/(2*N) * (dIdt[0:N,:] + dIdt[1:N+1,:])
    cont_dynR = R[1:N+1,:] - R[0:N,:] - T/(2*N) * (dRdt[0:N,:] + dRdt[1:N+1,:])

   # cont_dynV = V[1:N+1,:] - V[0:N,:] - T/(2*N) * (dVdt[0:N,:] + dVdt[1:N+1,:])


    cont_dyn = vertcat(
        reshape(cont_dynS,-1,1),
        reshape(cont_dynE,-1,1),
        reshape(cont_dynI,-1,1),
        reshape(cont_dynR,-1,1))

    ## Take into account the constraint  \sum_{j=1}^6 w_j(t) <= w_max
    #gg = vertcat(cont_dyn, sum2(u))
    #lower_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), w_min * np.ones(N+1))   # w_min=0
    #upper_bound_gg = vertcat(np.zeros(4 * num_age_groups * N), w_max * np.ones(N+1))
    
                        #print (S)
    ## Cost
   # print(     sum2(1-u)            )
    gg = vertcat(cont_dyn, sum2(u))
    lower_bound_gg = u_min * np.ones(N+1)   # w_min=0
    upper_bound_gg =  u_max * np.ones(N+1)
    cost_deaths = sum2(R[N, :] * death_rates*cost_per_death)     #   cost_deaths = sum1(mtimes(I,death_rates.T))
   # print(cost_deaths)
    # print(u)
    # print('between')
    # print(S[N, :])
    #print('break')
    #cost_lockdown = sum2(((1-u[N, :]))*cost_of_lockdown*tab_N) #Either tab_N or S
    cost_lockdown = sum2(  sum1(1-u) *cost_of_lockdown*tab_N  ) #Either tab_N or S
    #print(type(cost_lockdown))
    cost_all=cost_deaths+cost_lockdown
    ## Initial guess
    if init_S is None:
    	## [Manu] Below, we choose a very rough initialization, but it would be better to choose adequate "intuitive" controls
    	## (defined bang-bang "by hand") and generate the corresponding state variables S, E, I, R, V by Euler explicit method.
        init_S=mtimes(np.ones((N+1,1)), 2/3*initial_S)
        init_E=mtimes(np.ones((N+1,1)), 2/3*initial_E)
        init_I=mtimes(np.ones((N+1,1)), 2/3*initial_I)
        init_R=mtimes(np.ones((N+1,1)), 2/3*initial_R)

        #init_V=mtimes(np.ones((N+1,1)), 2/3*initial_V)
 
        init_u=(u_min+u_max)/2 * np.ones(N+1)
        ## [Manu] I have updated the rough way here to initialize w:
        #init_w=(w_min+w_max)/(2*num_age_groups) * (np.ones((N+1,num_age_groups)))
        
    #     init_w[:, 0:1] = 0
    init_xu=vertcat(
        reshape(init_S,-1,1),
        reshape(init_E,-1,1),
        reshape(init_I,-1,1),
        reshape(init_R,-1,1),
       # reshape(init_V,-1,1),

       # reshape(init_w,-1,1),
        init_u)

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

    # lower_bound_V = np.zeros((N+1,num_age_groups))
    # lower_bound_V[0,:] = initial_V
    # upper_bound_V = upper_bound * np.ones((N+1,num_age_groups))
    # upper_bound_V[0,:] = initial_V


    lower_bound_u = u_min*np.ones(N+1)
    upper_bound_u = u_max*np.ones(N+1)
   # lower_bound_w = w_min*np.ones((N+1,num_age_groups))
   # upper_bound_w = w_max*np.ones((N+1,num_age_groups))
   # upper_bound_w[:, 0:2] = 0   ## [Manu] this constraint means that we switch off the two first controls, right?

    lower_bound_xu = vertcat(
        reshape(lower_bound_S,-1,1),
        reshape(lower_bound_E,-1,1),
        reshape(lower_bound_I,-1,1),
        reshape(lower_bound_R,-1,1),
        lower_bound_u
        #reshape(lower_bound_V,-1,1),
        #reshape(lower_bound_w,-1,1),
        )
    upper_bound_xu = vertcat(
        reshape(upper_bound_S,-1,1),
        reshape(upper_bound_E,-1,1),
        reshape(upper_bound_I,-1,1),
        reshape(upper_bound_R,-1,1),
        #reshape(upper_bound_V,-1,1),
        #reshape(upper_bound_w,-1,1),
        upper_bound_u)
   
    ## Solve
    optim_problem = {
        'x' : vertcat(
            reshape(S, -1, 1),
            reshape(E, -1, 1),
            reshape(I, -1, 1),
            reshape(R, -1, 1),
           # reshape(V, -1, 1),
           # reshape(w, -1, 1),
            u,),
        'f' : cost_all,# You might change the cost here.
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
    # # options['ipopt.warm_start_init_point'] = 'yes'
    options['ipopt.max_iter'] = 10000
    # options['ipopt.expect_infeasible_problem'] = "no"
    # options['ipopt.bound_frac'] = 0.5
    # options['ipopt.start_with_resto'] = "no"
    # options['ipopt.required_infeasibility_reduction'] = 0.85
    # options['ipopt.acceptable_iter'] = 8
    solver = casadi.nlpsol('solver', 'ipopt', optim_problem, options)
    result = solver(x0=init_xu,                   ## initialization
                    lbx=lower_bound_xu,           ## lower bounds on the global unknown x
                    ubx=upper_bound_xu,          ## upper bounds on the global unknown x
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


    
    
    
    # Sv=xu[:(1 * num_age_groups * (N+1))]
    # Sv=reshape(S,N+1, num_age_groups)
    # Ev=xu[(1 * num_age_groups * (N+1)):(2 * num_age_groups * (N+1))]
    # Ev=reshape(E,N+1, num_age_groups)
    # Iv=xu[(2 * num_age_groups * (N+1)):(3 * num_age_groups * (N+1))]
    # Iv=reshape(I,N+1, num_age_groups)
    # Rv=xu[(3 * num_age_groups * (N+1)):(4 * num_age_groups * (N+1))]
    # Rv=reshape(R,N+1, num_age_groups)
    
    #w=xu[(4 * num_age_groups * (N+1)):(5 * num_age_groups * (N+1))] 
    #w=reshape(w,N+1,num_age_groups)
    #U=xu[(4 * num_age_groups * (N+1)):(4 * num_age_groups * (N+1)+(N+1))]
    u=xu[(4 * num_age_groups * (N+1)):]

    return S, E, I, R, u
