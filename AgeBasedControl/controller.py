#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:09:22 2023

@author: ryanweightman
"""

from casadi import *

from utils import Problem

"""
The first methods of this SuperClass are meant to be overriden, to be able to implement different
types of controls and dynamics in different SubClasses
"""

class ProblemSolver:
    def __init__(
        self,
        problem: Problem
    ):
        self.problem=problem
        ## Parameters
        self.T = self.problem.time_horizon
        self.N = self.problem.N
        self.contact_matrix = problem.contact_matrix
        self.gamma = problem.gamma
        # sigma = 1/60
        self.delta = problem.delta
        self.cost_of_lockdown=  70
        self.cost_of_lockdown_old=   45
        self.cost_of_lockdown_school = 30
        self.cost_per_death= 1500000
        self.u_min= problem.R0 * self.gamma
        self.u_max= problem.R0 * self.gamma  # bounds on u: if u_min=u_max then no lockdown
        self.upper_bound = inf
        self.beta = problem.R0 * self.gamma
        self.death_rates = problem.death_rates
        self.initial_S = problem.initial_S
        self.initial_E = problem.initial_E
        self.initial_I = problem.initial_I
        self.initial_R = problem.initial_R

        self.model_name="former_model"

        self.tab_N= problem.population
        self.num_age_groups = self.tab_N.shape[1]

        self.Ntot=sum2(self.tab_N)

    def interaction_matrices(self):
        """Define interaction matrices"""
        # Must be defined in subclasses according to the choice of the controls
        return None

    def model_dynamics(self,S,E,I,interaction_matrices,controls):
        """Defines the dynamic of the model"""
        # Same thing
        return None

    def cost(self, I, R, interaction_matrices,controls):
        """Defines the cost function"""
        # Same thing
        return None

    def solve_control_problem(self, init_S=None,
                            init_E=None, init_I=None, init_R=None, init_V=None,
                            init_w=None, init_u=None):
        ## Declaration of variables
        S=MX.sym('S', self.N+1, self.num_age_groups)
        E=MX.sym('E', self.N+1, self.num_age_groups)
        I=MX.sym('I', self.N+1, self.num_age_groups)
        R=MX.sym('R', self.N+1, self.num_age_groups)
    # V=MX.sym('V',N+1,num_age_groups)
        dSdt=MX.sym('dSdt', self.N+1, self.num_age_groups)
        dEdt=MX.sym('dEdt', self.N+1, self.num_age_groups)
        dIdt=MX.sym('dIdt', self.N+1, self.num_age_groups)
        dRdt=MX.sym('dRdt', self.N+1, self.num_age_groups)

        u=MX.sym('u',self.N+1)
        w=MX.sym('w',self.N+1, self.numControls)

    ################################### Building Matrices for age based control
        interaction_matrices=self.interaction_matrices()
    #####################################


        ## Discretization of dynamics with implicit RK2
        dSdt,dEdt,dIdt,dRdt=self.model_dynamics(S,E,I,interaction_matrices, w)

        cont_dynS = S[1:self.N+1,:] - S[0:self.N,:] - self.T/(2*self.N) * (dSdt[0:self.N,:] + dSdt[1:self.N+1,:])
        cont_dynE = E[1:self.N+1,:] - E[0:self.N,:] - self.T/(2*self.N) * (dEdt[0:self.N,:] + dEdt[1:self.N+1,:])
        cont_dynI = I[1:self.N+1,:] - I[0:self.N,:] - self.T/(2*self.N) * (dIdt[0:self.N,:] + dIdt[1:self.N+1,:])
        cont_dynR = R[1:self.N+1,:] - R[0:self.N,:] - self.T/(2*self.N) * (dRdt[0:self.N,:] + dRdt[1:self.N+1,:])




        cont_dyn = vertcat(
            reshape(cont_dynS,-1,1),
            reshape(cont_dynE,-1,1),
            reshape(cont_dynI,-1,1),
            reshape(cont_dynR,-1,1))


        ## Take into account the constraint  \sum_{j=1}^6 w_j(t) <= w_max
        gg, lower_bound_gg, upper_bound_gg = self.gg_and_bounds(cont_dyn, w)

        cost_all = self.cost(I, R, interaction_matrices,w)


        ## Initial guess
        if init_S is None:

            init_S=mtimes(np.ones((self.N+1,1)), 2/3*self.initial_S)
            init_E=mtimes(np.ones((self.N+1,1)), 2/3*self.initial_E)
            init_I=mtimes(np.ones((self.N+1,1)), 2/3*self.initial_I)
            init_R=mtimes(np.ones((self.N+1,1)), 2/3*self.initial_R)


            init_u=(self.u_min+self.u_max)/2 * np.ones(self.N+1)
            init_w=.5 * (np.ones((self.N+1,self.numControls)))

        init_xu=vertcat(
            reshape(init_S,-1,1),
            reshape(init_E,-1,1),
            reshape(init_I,-1,1),
            reshape(init_R,-1,1),
            reshape(init_w,-1,1),
            init_u)

        lower_bound_S = np.zeros((self.N+1,self.num_age_groups))
        lower_bound_S[0,:] = self.initial_S
        upper_bound_S =  self.upper_bound * np.ones((self.N+1,self.num_age_groups))
        upper_bound_S[0,:] = self.initial_S

        lower_bound_E = np.zeros((self.N+1,self.num_age_groups))
        lower_bound_E[0,:]= self.initial_E
        upper_bound_E = self.upper_bound * np.ones((self.N+1,self.num_age_groups))
        upper_bound_E[0,:] = self.initial_E

        lower_bound_I = np.zeros((self.N+1,self.num_age_groups))
        lower_bound_I[0,:] = self.initial_I
        upper_bound_I = self.upper_bound * np.ones((self.N+1,self.num_age_groups))
        upper_bound_I[0,:] = self.initial_I

        lower_bound_R =  np.zeros((self.N+1,self.num_age_groups))
        lower_bound_R[0,:] = self.initial_R
        upper_bound_R = self.upper_bound * np.ones((self.N+1,self.num_age_groups))
        upper_bound_R[0,:] = self.initial_R



        lower_bound_u = self.u_min*np.ones(self.N+1)
        upper_bound_u = self.u_max*np.ones(self.N+1)
        lower_bound_w = self.w_min*np.ones((self.N+1,self.numControls))
        upper_bound_w = self.w_max*np.ones((self.N+1,self.numControls))

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

        S=xu[:(1 * self.num_age_groups * (self.N+1))]
        S=reshape(S,self.N+1, self.num_age_groups)
        E=xu[(1 * self.num_age_groups * (self.N+1)):(2 * self.num_age_groups * (self.N+1))]
        E=reshape(E,self.N+1, self.num_age_groups)
        I=xu[(2 * self.num_age_groups * (self.N+1)):(3 * self.num_age_groups * (self.N+1))]
        I=reshape(I,self.N+1, self.num_age_groups)
        R=xu[(3 * self.num_age_groups * (self.N+1)):(4 * self.num_age_groups * (self.N+1))]
        R=reshape(R,self.N+1, self.num_age_groups)

        w=xu[(4 * self.num_age_groups * (self.N+1)):(4 * self.num_age_groups * (self.N+1))+(self.numControls*(self.N+1))]
        w=reshape(w,self.N+1,self.numControls)
        u=xu[((4 * self.num_age_groups * (self.N+1))+self.numControls * (self.N+1)):]
        print(cost)
        return S, E, I, R, w,cost
