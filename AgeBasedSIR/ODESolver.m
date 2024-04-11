function dYdt = ODESolver(t,Y,n, beta, gamma)
 lOI = 1/180; %% 1/number of days immune- loss of immunity rate

S = Y(1:n);   %% Susceptibles
I = Y(n+1:(2*n)) ;    %infected populations governed by choice of n
R = Y(2*n+1);   %% Recovered
N = sum(S)+ sum(I)+ R;%% Total Population

for k = 1:100
  Ihat = repmat(I, 1, n);
end
Ihat = Ihat';

for i = 1:n
    for j = 1:n
        if abs(j-i)>20
            Ihat(i,j) = 0;
        end
    end

end

dSdt = - (S.*beta.*sum(Ihat,2)); % evolution of susceptible. for seasonal add "+lOI*R" and define loI- loss of immunity rate
dIdt = (S.*beta.*sum(Ihat,2)) - gamma .* I ;% evolution of Infected populations (matrix form)
dRdt = sum(gamma .* I); % Recovered for seasonal add "-lOI*R"
dYdt = [dSdt ;  dIdt; dRdt];% Solution matrix
