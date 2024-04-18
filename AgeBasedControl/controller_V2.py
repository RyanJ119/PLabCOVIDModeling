import numpy as np
from casadi import *

from controller import ProblemSolver
from utils import Problem

class ProblemSolver2(ProblemSolver):
    def __init__(self, problem: Problem):
        self.numControls = 3

        super().__init__(problem)
        self.tau1 = 4/5
        self.tau2 = 2/3
        self.tau3 = 1/7*1/5 # 1/7 is approximately half the percentage of adults who have children, 1/5 is a guess for the sub proportion of these adult unable to afford chidcare
        self.w_min=[0.,0.,0.]
        self.w_max=[.6,.8,1.]
        self.model_name="V2"

    def interaction_matrices(self):
        """Define interaction matrices"""
        dimensions = np.shape(self.contact_matrix)
        rows, columns = dimensions

        mat_old= self.contact_matrix.copy()

        for i in range(rows):
            for j in range(columns):
                mat_old[i][j] = mat_old[i][j]*self.tau1
                if i<rows-1:
                    if j<columns-1:
                        mat_old[i][j]=0

        mat_school=self.contact_matrix.copy()

        for i in range(rows):
            for j in range(columns):
                mat_school[i][j] = mat_school[i][j]*self.tau2
                if (i > 3 or j > 3):
                    mat_school[i][j]=0
        mat_public=self.contact_matrix.copy()-mat_school.copy()-mat_old.copy()

        mat_parent_impacted_school_closure=self.tau3*mat_public.copy()

        for i in range(rows):
            for j in range(columns):
                if (i>8 or j>8) or (i < 4 and j < 4) or (4 <= i and 4 <= j):
                    mat_parent_impacted_school_closure[i][j]=0

        return [mat_old, mat_school, mat_public-mat_parent_impacted_school_closure, mat_parent_impacted_school_closure]

    def model_dynamics(self,dSdt,dEdt,dIdt,dRdt,S,E,I,R,interaction_matrices,controls):
        """Defines the dynamic of the model"""
        dSdt= -1* self.beta * S * (1-controls[:,2]) * ( (1-controls[:,0]) * mtimes(I,interaction_matrices[0]) + (1-controls[:,1]) * mtimes(I,interaction_matrices[1]+interaction_matrices[3]) + mtimes(I,interaction_matrices[2]) )/ repmat(mtimes(self.tab_N, self.contact_matrix), self.N+1, 1)
        dEdt= self.beta * S * (1-controls[:,2]) * ( (1-controls[:,0]) * mtimes(I,interaction_matrices[0]) + (1-controls[:,1]) * mtimes(I,interaction_matrices[1]+interaction_matrices[3]) + mtimes(I,interaction_matrices[2]) )/ repmat(mtimes(self.tab_N, self.contact_matrix), self.N+1, 1) - self.delta * E
        dIdt= self.delta * E - self.gamma * I
        dRdt= self.gamma * I
        return dSdt,dEdt,dIdt,dRdt

    def cost(self, I, R, interaction_matrices,controls):
        """Defines the cost function"""
        cost_deaths = sum2(R[self.N, :] * self.death_rates)*self.cost_per_death

        cost_lockdown=sum2((sum1( interaction_matrices[0]) / sum1(self.contact_matrix)) *self.tab_N )*sum1(self.cost_of_lockdown_old*sum2(controls[:,0]+controls[:,2]-controls[:,0]*controls[:,2])) + sum2((sum1( interaction_matrices[1]) / sum1(self.contact_matrix)) *self.tab_N )*sum1(self.cost_of_lockdown_school*sum2(controls[:,1]+controls[:,2]-controls[:,1]*controls[:,2]))+ sum2((sum1( interaction_matrices[2]) / sum1(self.contact_matrix)) *self.tab_N )*sum1(self.cost_of_lockdown*sum2(controls[:,2]))+sum2((sum1( interaction_matrices[3]) / sum1(self.contact_matrix)) *self.tab_N )*sum1(self.cost_of_lockdown*sum2(controls[:,1]+controls[:,2]-controls[:,1]*controls[:,2]))

        cost_end = sum2(I[self.N, :] * self.death_rates)*self.cost_per_death*90

        cost_all=cost_deaths+cost_lockdown+cost_end

        return cost_all
    
    def bounds_gg(self, cont_dyn, w):
        gg = vertcat(cont_dyn, w[:,0], w[:,1], w[:,2])
        lower_bound_gg = vertcat(np.zeros(4 * self.num_age_groups * self.N), np.concatenate((self.w_min[0] * np.ones(self.N+1),self.w_min[1] * np.ones(self.N+1),self.w_min[2] * np.ones(self.N+1)),axis=None))   # w_min=0
        upper_bound_gg = vertcat(np.zeros(4 * self.num_age_groups * self.N), np.concatenate((self.w_max[0] * np.ones(self.N+1),self.w_max[1] * np.ones(self.N+1),self.w_max[2] * np.ones(self.N+1)),axis=None))
        return gg, lower_bound_gg, upper_bound_gg
