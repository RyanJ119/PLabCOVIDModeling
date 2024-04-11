import numpy as np
from casadi import *

from controller import ProblemSolver
from utils import Problem

class ProblemSolver1(ProblemSolver):
    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.tau1 = 1#4/5
        self.tau2 = 1#2/3
        self.numControls = 3
    
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
                if (i != 0 and i!=1 and i!=2 and i!=3 ) or (j != 0 and j!=1 and j!=2 and j!=3):
                    mat_school[i][j]=0
        matrix4=self.contact_matrix.copy()
            
        return [mat_old, mat_school, matrix4]
    
    def model_dynamics(self,S,E,I,R,interaction_matrices,controls):
        """Defines the dynamic of the model"""
        return ( -1*( ((1-controls[:,0]) * self.beta * S * (mtimes(I,interaction_matrices[0])) ) +((1-controls[:,1]) * self.beta * S * (mtimes(I,interaction_matrices[1]))) +((1-controls[:,2]) * self.beta * S * (mtimes(I,interaction_matrices[2]))) )/ repmat(mtimes(self.tab_N, self.contact_matrix), self.N+1, 1) ), ( ( ((1-controls[:,0]) * self.beta * S * (mtimes(I,interaction_matrices[0])))  + ((1-controls[:,1]) * self.beta * S * (mtimes(I,interaction_matrices[1]))) +((1-controls[:,2]) * self.beta * S * (mtimes(I,interaction_matrices[2])) ))/ repmat(mtimes(self.tab_N, self.contact_matrix), self.N+1, 1) )  - self.delta * E, self.delta * E - self.gamma * I, self.gamma * I
        
    def cost(self, I, R, interaction_matrices,controls):
        """Defines the cost function"""
        cost_deaths = sum2(R[self.N, :] * self.death_rates)*self.cost_per_death
        
        cost_lockdown=sum2((sum1( interaction_matrices[0]) / sum1(self.contact_matrix)) *self.tab_N )*sum1(self.cost_of_lockdown_old*sum2(controls[:,0])) + sum2((sum1( interaction_matrices[1]) / sum1(self.contact_matrix)) *self.tab_N )*sum1(self.cost_of_lockdown_school*sum2(controls[:,1]))+ sum2((sum1( interaction_matrices[2]) / sum1(self.contact_matrix)) *self.tab_N )*sum1(self.cost_of_lockdown*sum2(controls[:,2]))

        cost_end = sum2(I[self.N, :] * self.death_rates)*self.cost_per_death*90
        
        cost_all=cost_deaths+cost_lockdown+cost_end
        
        return cost_all