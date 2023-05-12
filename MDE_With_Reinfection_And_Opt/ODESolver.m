function dYdt = ODESolver(t,Y,n, beta, gamma,u, betaHat, sigma,eta,d)
 %lOI = 1/180; %% 1/number of days immune- loss of immunity rate
 
S = Y(1);   %% Susceptibles
I = Y(2:(n+1))  ;    %infected populations governed by choice of n

R = Y( (n+2) : 2*n+1 );   %% Recovered

Sr = Y( 2*n+2 : 3*n+1 ); 
H = Y( 3*n+2 : 4*n+1 ); 
D = Y(4*n+2);

N = S+ sum(I)+ sum(R)+sum(Sr)+sum(H)+D; %% Total Population

dSdt = -sum(beta * (S/N) .* I); % evolution of susceptible. for seasonal add "+lOI*R" and define loI- loss of immunity rate
dIdt = beta * (S/N) .*  I + sum((betaHat.* I'.*Sr), 2)./N - gamma .*  I  - .0003.* I - eta .* I;% evolution of Infected populations (matrix form)
dRdt = gamma .*  I - sigma.*R; % Recovered for seasonal add "-lOI*R"
Srdt =  sigma.*R - sum((betaHat.* I'.*Sr), 2)./N ;   % mtimes( betaHat, Sr' )/N .* I'
dHdt = eta .* I  - d*H;
dDdt = sum(.0003.* I)+sum(d*H);
dYdt = [dSdt ;  dIdt; dRdt; Srdt; dHdt; dDdt ];% Solution matrix





