import numpy as np
from casadi import *

from controller import ProblemSolver
from utils import Problem

class ProblemSolver1(ProblemSolver):
    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.numControls = 3
        self.w_min=[0.,0.,0.]
        self.w_max=[.6,.8,1.]
        self.tau1 = 4/5
        self.tau2 = 2/3
        self.model_name="V1"

    def interaction_matrices(self):
        """Define interaction matrices"""
        dimensions = np.shape(self.contact_matrix)
        rows, columns = dimensions

        mat_old= self.tau1*(self.contact_matrix.copy())

        for i in range(rows):
            for j in range(columns):
                if i < rows-1 and j < columns-1:
                    mat_old[i][j]=0 # we get rid of the interactions that don't involve the elderly

        mat_school=self.tau2*(self.contact_matrix.copy())

        for i in range(rows):
            for j in range(columns):
                if i > 3 or j > 3:
                    mat_school[i][j]=0 # we get rid of the interactions other than children/children

        mat_public=self.contact_matrix.copy()-mat_school.copy()-mat_old.copy() # remaining of the interactions

        return [mat_old, mat_school, mat_public]

    def model_dynamics(self,S,E,I,interaction_matrices,controls):
        """Defines the dynamic of the model"""
        dSdt= -1* self.beta * S *( (1-controls[:,0]) * mtimes(I,interaction_matrices[0]) + (1-controls[:,1]) * mtimes(I,interaction_matrices[1]) +(1-controls[:,2]) * mtimes(I,interaction_matrices[2]) )/ repmat(mtimes(self.tab_N, self.contact_matrix), self.N+1, 1)
        dEdt= self.beta * S *( (1-controls[:,0]) * mtimes(I,interaction_matrices[0]) + (1-controls[:,1]) * mtimes(I,interaction_matrices[1]) +(1-controls[:,2]) * mtimes(I,interaction_matrices[2]) )/ repmat(mtimes(self.tab_N, self.contact_matrix), self.N+1, 1) - self.delta * E
        dIdt= self.delta * E - self.gamma * I
        dRdt= self.gamma * I
        return dSdt,dEdt,dIdt,dRdt

    def cost(self, I, R, interaction_matrices,controls):
        """Defines the cost function"""
        cost_deaths = sum2(R[self.N, :] * self.death_rates)*self.cost_per_death

        cost_lockdown=sum2((sum1( interaction_matrices[0]) / sum1(self.contact_matrix)) *self.tab_N )*sum1(self.cost_of_lockdown_old*sum2(controls[:,0])) + sum2((sum1( interaction_matrices[1]) / sum1(self.contact_matrix)) *self.tab_N )*sum1(self.cost_of_lockdown_school*sum2(controls[:,1]))+ sum2((sum1( interaction_matrices[2]) / sum1(self.contact_matrix)) *self.tab_N )*sum1(self.cost_of_lockdown*sum2(controls[:,2]))

        cost_end = sum2(I[self.N, :] * self.death_rates)*self.cost_per_death*90

        cost_all=cost_deaths+cost_lockdown+cost_end

        return cost_all

    def gg_and_bounds(self, cont_dyn, w):
        gg = vertcat(cont_dyn, w[:,0], w[:,1], w[:,2])
        lower_bound_gg = vertcat(np.zeros(4 * self.num_age_groups * self.N), np.concatenate((self.w_min[0] * np.ones(self.N+1),self.w_min[1] * np.ones(self.N+1),self.w_min[2] * np.ones(self.N+1)),axis=None))   # w_min=0
        upper_bound_gg = vertcat(np.zeros(4 * self.num_age_groups * self.N), np.concatenate((self.w_max[0] * np.ones(self.N+1),self.w_max[1] * np.ones(self.N+1),self.w_max[2] * np.ones(self.N+1)),axis=None))
        return gg, lower_bound_gg, upper_bound_gg
