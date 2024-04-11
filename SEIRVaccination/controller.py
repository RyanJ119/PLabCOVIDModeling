from casadi import *
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import os
from datetime import datetime
import math


def solve_control_problem(problem, max_num_vaccines_per_day, init_S=None,
                          init_E=None, init_I=None, init_R=None, init_Sv=None,
                          init_Ev=None, init_Iv=None, init_Rv=None, init_w=None, init_u=None):
    ## Parameters
    T = problem.time_horizon
    N = problem.N
    a = problem.contact_matrix
    gamma = problem.gamma
    delta = problem.delta
    u_min= problem.R0 * gamma
    u_max= problem.R0 * gamma  # bounds on u: if u_min=u_max then no lockdown
    upper_bound = inf
    w_min=0
    w_max=max_num_vaccines_per_day  ## w_max is the upper threshold on w(t) = \sum_{j=1}^6 w_j(t)  (maximal number of vaccines per day)
    death_rates = problem.death_rates
    initial_S = problem.initial_S
    initial_E = problem.initial_E
    initial_I = problem.initial_I
    initial_R = problem.initial_R
    #initial_V = problem.initial_V

    initial_Sv = problem.initial_Sv
    initial_Ev = problem.initial_Ev
    initial_Iv = problem.initial_Iv
    initial_Rv = problem.initial_Rv




    tab_N= problem.population
    num_age_groups = tab_N.shape[1]

    Ntot=sum2(tab_N)  # total population
    #initial_V =  np.zeros((1,num_age_groups))
    initial_Sv =  np.zeros((1,num_age_groups))
    initial_Ev =  np.zeros((1,num_age_groups))
    initial_Iv =  np.zeros((1,num_age_groups))
    initial_Rv =  np.zeros((1,num_age_groups))
    ## Declaration of variables
    S=MX.sym('S', N+1, num_age_groups)
    E=MX.sym('E', N+1, num_age_groups)
    I=MX.sym('I', N+1, num_age_groups)
    R=MX.sym('R', N+1, num_age_groups)
   # V=MX.sym('V',N+1,num_age_groups)
    Sv=MX.sym('Sv', N+1, num_age_groups)
    Ev=MX.sym('Ev', N+1, num_age_groups)
    Iv=MX.sym('Iv', N+1, num_age_groups)
    Rv=MX.sym('Rv', N+1, num_age_groups)
    dSdt=MX.sym('dSdt', N+1, num_age_groups)
    dEdt=MX.sym('dEdt', N+1, num_age_groups)
    dIdt=MX.sym('dIdt', N+1, num_age_groups)
    dRdt=MX.sym('dRdt', N+1, num_age_groups)
    #dVdt=MX.sym('dVdt', N+1, num_age_groups)
    dSvdt=MX.sym('dSvdt', N+1, num_age_groups)
    dEvdt=MX.sym('dEvdt', N+1, num_age_groups)
    dIvdt=MX.sym('dIvdt', N+1, num_age_groups)
    dRvdt=MX.sym('dRvdt', N+1, num_age_groups)
    u=MX.sym('u',N+1)
    w=MX.sym('w',N+1,num_age_groups)


    ## Discretization of dynamics with implicit RK2 (S+E+I)

    dSdt = -u * S * ((mtimes(I,a)+ .3 * mtimes(Iv,a)) / repmat(mtimes(tab_N, a), N+1, 1)) - w * S/(S+E+I) #*(S>=0)*(E>=0)*(I>=0)

    dEdt =  u * S * ((mtimes(I,a)+ .3 * mtimes(Iv,a)) / repmat(mtimes(tab_N, a), N+1, 1))  - delta * E- w * E/(S+E+I)

    dIdt = delta * E - gamma * I - w * I/(S+E+I)

    dRdt = gamma * I

   # dVdt = w #*(S>=0)*(E>=0)*(I>=0)

    dSvdt = -u * Sv * ((.3 * mtimes(I,a)+ .09 * mtimes(Iv,a)) / repmat(mtimes(tab_N, a), N+1, 1)) + w * S/(S+E+I) #*(S>=0)*(E>=0)*(I>=0)

    dEvdt =  u * Sv * ((.3 * mtimes(I,a)+ .09 * mtimes(Iv,a)) / repmat(mtimes(tab_N, a), N+1, 1))  - delta * Ev + w * E/(S+E+I)

    dIvdt = delta * Ev - gamma * Iv + w * I/(S+E+I)

    dRvdt = gamma * Iv





    cont_dynS = S[1:N+1,:] - S[0:N,:] - T/(2*N) * (dSdt[0:N,:] + dSdt[1:N+1,:])
    cont_dynE = E[1:N+1,:] - E[0:N,:] - T/(2*N) * (dEdt[0:N,:] + dEdt[1:N+1,:])
    cont_dynI = I[1:N+1,:] - I[0:N,:] - T/(2*N) * (dIdt[0:N,:] + dIdt[1:N+1,:])
    cont_dynR = R[1:N+1,:] - R[0:N,:] - T/(2*N) * (dRdt[0:N,:] + dRdt[1:N+1,:])

   # cont_dynV = V[1:N+1,:] - V[0:N,:] - T/(2*N) * (dVdt[0:N,:] + dVdt[1:N+1,:])
    cont_dynSv = Sv[1:N+1,:] - Sv[0:N,:] - T/(2*N) * (dSvdt[0:N,:] + dSvdt[1:N+1,:])
    cont_dynEv = Ev[1:N+1,:] - Ev[0:N,:] - T/(2*N) * (dEvdt[0:N,:] + dEvdt[1:N+1,:])
    cont_dynIv = Iv[1:N+1,:] - Iv[0:N,:] - T/(2*N) * (dIvdt[0:N,:] + dIvdt[1:N+1,:])
    cont_dynRv = Rv[1:N+1,:] - Rv[0:N,:] - T/(2*N) * (dRvdt[0:N,:] + dRvdt[1:N+1,:])

    cont_dyn = vertcat(
        reshape(cont_dynS,-1,1),
        reshape(cont_dynE,-1,1),
        reshape(cont_dynI,-1,1),
        reshape(cont_dynR,-1,1),
        #reshape(cont_dynV,-1,1)
        reshape(cont_dynSv,-1,1),
        reshape(cont_dynEv,-1,1),
        reshape(cont_dynIv,-1,1),
        reshape(cont_dynRv,-1,1))

    ## Take into account the constraint  \sum_{j=1}^6 w_j(t) <= w_max
    gg = vertcat(cont_dyn, sum2(w))
    lower_bound_gg = vertcat(np.zeros(8 * num_age_groups * N), w_min * np.ones(N+1))   # w_min=0
    upper_bound_gg = vertcat(np.zeros(8 * num_age_groups * N), w_max * np.ones(N+1))
                        #print (S)
    ## Cost
    cost_deaths = sum2(R[N, :] * death_rates)     #   cost_deaths = sum1(mtimes(I,death_rates.T))

    ## Initial guess
    if init_S is None:
    	## [Manu] Below, we choose a very rough initialization, but it would be better to choose adequate "intuitive" controls
    	## (defined bang-bang "by hand") and generate the corresponding state variables S, E, I, R, V by Euler explicit method.
        init_S=mtimes(np.ones((N+1,1)), 2/3*initial_S)
        init_E=mtimes(np.ones((N+1,1)), 2/3*initial_E)
        init_I=mtimes(np.ones((N+1,1)), 2/3*initial_I)
        init_R=mtimes(np.ones((N+1,1)), 2/3*initial_R)
        #init_V=mtimes(np.ones((N+1,1)), 2/3*initial_V)
        init_Sv=mtimes(np.ones((N+1,1)), 2/3*initial_Sv)
        init_Ev=mtimes(np.ones((N+1,1)), 2/3*initial_Ev)
        init_Iv=mtimes(np.ones((N+1,1)), 2/3*initial_Iv)
        init_Rv=mtimes(np.ones((N+1,1)), 2/3*initial_Rv)
        init_u=(u_min+u_max)/2 * np.ones(N+1)
        ## [Manu] I have updated the rough way here to initialize w:
        init_w=(w_min+w_max)/(2*num_age_groups) * (np.ones((N+1,num_age_groups)))

    #     init_w[:, 0:1] = 0
    init_xu=vertcat(
        reshape(init_S,-1,1),
        reshape(init_E,-1,1),
        reshape(init_I,-1,1),
        reshape(init_R,-1,1),
       # reshape(init_V,-1,1),
        reshape(init_Sv,-1,1),
        reshape(init_Ev,-1,1),
        reshape(init_Iv,-1,1),
        reshape(init_Rv,-1,1),
        reshape(init_w,-1,1),
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
    lower_bound_Sv = np.zeros((N+1,num_age_groups))
    lower_bound_Sv[0,:] = initial_Sv
    upper_bound_Sv =  upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_Sv[0,:] = initial_Sv

    lower_bound_Ev = np.zeros((N+1,num_age_groups))
    lower_bound_Ev[0,:]= initial_Ev
    upper_bound_Ev = upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_Ev[0,:] = initial_Ev

    lower_bound_Iv = np.zeros((N+1,num_age_groups))
    lower_bound_Iv[0,:] = initial_Iv
    upper_bound_Iv = upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_Iv[0,:] = initial_Iv

    lower_bound_Rv =  np.zeros((N+1,num_age_groups))
    lower_bound_Rv[0,:] = initial_Rv
    upper_bound_Rv = upper_bound * np.ones((N+1,num_age_groups))
    upper_bound_Rv[0,:] = initial_Rv

    lower_bound_u = u_min*np.ones(N+1)
    upper_bound_u = u_max*np.ones(N+1)
    lower_bound_w = w_min*np.ones((N+1,num_age_groups))
    upper_bound_w = w_max*np.ones((N+1,num_age_groups))
    upper_bound_w[:, 0:2] = 0   ## [Manu] this constraint means that we switch off the two first controls, right?

    lower_bound_xu = vertcat(
        reshape(lower_bound_S,-1,1),
        reshape(lower_bound_E,-1,1),
        reshape(lower_bound_I,-1,1),
        reshape(lower_bound_R,-1,1),
        #reshape(lower_bound_V,-1,1),
        reshape(lower_bound_Sv,-1,1),
        reshape(lower_bound_Ev,-1,1),
        reshape(lower_bound_Iv,-1,1),
        reshape(lower_bound_Rv,-1,1),
        reshape(lower_bound_w,-1,1),
        lower_bound_u)
    upper_bound_xu = vertcat(
        reshape(upper_bound_S,-1,1),
        reshape(upper_bound_E,-1,1),
        reshape(upper_bound_I,-1,1),
        reshape(upper_bound_R,-1,1),
        #reshape(upper_bound_V,-1,1),
        reshape(upper_bound_Sv,-1,1),
        reshape(upper_bound_Ev,-1,1),
        reshape(upper_bound_Iv,-1,1),
        reshape(upper_bound_Rv,-1,1),
        reshape(upper_bound_w,-1,1),
        upper_bound_u)

    ## Solve
    optim_problem = {
        'x' : vertcat(
            reshape(S, -1, 1),
            reshape(E, -1, 1),
            reshape(I, -1, 1),
            reshape(R, -1, 1),
           # reshape(V, -1, 1),
            reshape(Sv, -1, 1),
            reshape(Ev, -1, 1),
            reshape(Iv, -1, 1),
            reshape(Rv, -1, 1),
            reshape(w, -1, 1),
            u,),
        'f' : cost_deaths, # You might change the cost here.
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
    Sv=xu[(4 * num_age_groups * (N+1)):(5 * num_age_groups * (N+1))]
    Sv=reshape(Sv
               ,N+1, num_age_groups)
    Ev=xu[(5 * num_age_groups * (N+1)):(6 * num_age_groups * (N+1))]
    Ev=reshape(Ev,N+1, num_age_groups)
    Iv=xu[(6 * num_age_groups * (N+1)):(7 * num_age_groups * (N+1))]
    Iv=reshape(Iv,N+1, num_age_groups)
    Rv=xu[(7 * num_age_groups * (N+1)):(8 * num_age_groups * (N+1))]
    Rv=reshape(Rv,N+1, num_age_groups)



    # Sv=xu[:(1 * num_age_groups * (N+1))]
    # Sv=reshape(S,N+1, num_age_groups)
    # Ev=xu[(1 * num_age_groups * (N+1)):(2 * num_age_groups * (N+1))]
    # Ev=reshape(E,N+1, num_age_groups)
    # Iv=xu[(2 * num_age_groups * (N+1)):(3 * num_age_groups * (N+1))]
    # Iv=reshape(I,N+1, num_age_groups)
    # Rv=xu[(3 * num_age_groups * (N+1)):(4 * num_age_groups * (N+1))]
    # Rv=reshape(R,N+1, num_age_groups)

    w=xu[(8 * num_age_groups * (N+1)):(9 * num_age_groups * (N+1))]
    w=reshape(w,N+1,num_age_groups)
    u=xu[(9 * num_age_groups * (N+1)):]

    return S, E, I, R, Sv,Ev,Iv,Rv, w, u
