import numpy as np

def simulate(
    problem,
    w,
):
    T = problem.time_horizon
    N = problem.N
    a = problem.contact_matrix
    gamma = problem.gamma
    delta = problem.delta
    u_max = problem.R0 * gamma
    tab_N = problem.population
    num_age_groups = problem.num_age_groups
    u = u_max * np.ones((N + 1, 1))

    S = np.zeros((N + 1, num_age_groups))
    S[0, :] = problem.initial_S
    E = np.zeros((N + 1, num_age_groups))
    E[0, :] = problem.initial_E
    I = np.zeros((N + 1, num_age_groups))
    I[0, :] = problem.initial_I
    R = np.zeros((N + 1, num_age_groups))
    R[0, :] = problem.initial_R
    
    Sv = np.zeros((N + 1, num_age_groups))
    Sv[0, :] = problem.initial_Sv
    Ev = np.zeros((N + 1, num_age_groups))
    Ev[0, :] = problem.initial_Ev
    Iv = np.zeros((N + 1, num_age_groups))
    Iv[0, :] = problem.initial_Iv
    Rv = np.zeros((N + 1, num_age_groups))
    Rv[0, :] = problem.initial_Rv
    
    for k in range(N):
       # print((S[k, :]+E[k, :]+I[k, :]))
        S[k +1, :] = np.fmax(

            S[k, :]
            + T / N * (-u[k] * S[k, :] * (((I[k, :] @ a)+(0.3*(Iv[k, :] @ a) )) / (tab_N @ a)) - np.multiply(w[k, :], np.divide(S[k, :],(S[k, :]+E[k, :]+I[k, :]),out=np.zeros_like(S[k, :]),where=(S[k, :]+E[k, :]+I[k, :])!=0  ))),
            0,
        )
        #np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        E[k + 1, :] = np.fmax(
            E[k, :]
            + T/ N * (u[k] * S[k, :] * (((I[k, :] @ a)+(0.3*(Iv[k, :] @ a) )) / (tab_N @ a)) - (delta * E[k, :])- np.multiply(w[k, :], np.divide(E[k, :],(S[k, :]+E[k, :]+I[k, :]),  out=np.zeros_like(E[k, :]), where=(S[k, :]+E[k, :]+I[k, :])!=0))),
            0,
        )
        I[k + 1, :] = np.fmax(
            I[k, :] + T / N * (delta * E[k + 1, :] - gamma * I[k, :] - np.multiply(w[k, :],np.divide(I[k, :],(S[k, :]+E[k, :]+I[k, :]), out=np.zeros_like(I[k, :]), where=(S[k, :]+E[k, :]+I[k, :])!=0))), 0
        )
        R[k + 1, :] = np.fmax(R[k, :] + T / N * gamma * I[k + 1, :], 0)
       
        
        Sv[k + 1, :] = np.fmax(
            Sv[k, :]
            + T / N * (-u[k] * Sv[k, :] * ((.3*(I[k, :] @ a)+(0.09*(Iv[k, :] @ a) )) / (tab_N @ a)) 
                       + np.multiply(w[k, :], np.divide(S[k, :],(S[k, :]+E[k, :]+I[k, :]), out=np.zeros_like(S[k, :]), where=(S[k, :]+E[k, :]+I[k, :])!=0))),
            0,
        )
        Ev[k + 1, :] = np.fmax(
            Ev[k, :]
            + T
            / N
            * ((u[k] * Sv[k, :] *  ((0.3*(I[k, :] @ a)+(0.09*(Iv[k, :] @ a) )) / (tab_N @ a)) - delta * Ev[k, :])+ np.multiply(w[k, :],np.divide(E[k, :],(S[k, :]+E[k, :]+I[k, :]), out=np.zeros_like(E[k, :]), where=(S[k, :]+E[k, :]+I[k, :])!=0))),
            0,
        )
        Iv[k + 1, :] = np.fmax(
            Iv[k, :] + T / N * ((delta * Ev[k + 1, :] - gamma * Iv[k, :])+ np.multiply(w[k, :],np.divide((I[k, :]),(S[k, :]+E[k, :]+I[k, :]), out=np.zeros_like(I[k, :]), where=(S[k, :]+E[k, :]+I[k, :])!=0))), 0
        )
        Rv[k + 1, :] = np.fmax(Rv[k, :] + T / N * gamma * Iv[k + 1, :], 0)
        
       
        
       
        
       
        
       
        
       
        

    return S, E, I, R, Sv, Ev, Iv, Rv
