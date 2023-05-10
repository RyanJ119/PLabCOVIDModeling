clear all;

Inputs; %bring in all chosen input values

%% Call the solver i times 

    tRange = 0:1:totalDays;  
    

    
    
    
    [tSol,YSol] = ode45(@(t,Y) ODESolver(t,Y, n, beta', gamma'), tRange, Yo);
  
    %%%% concatinating matrices double counts last previous entry/first new
    %%%% entry. Delete one of them here, then concatonate old solution with new
    %%%% solution
    
    if length(S)>0
        S(end,:)=[];
    end
    
    S = vertcat(S, YSol(:,1:n));
    
    if length(I)>0
        I(end, :) = [] ;
    end
    
    I = vertcat(I,  YSol(:,n+1:2*n));
    
    if length(R)>0
        R (end)=[];
    end
    R = vertcat(R, YSol(:,(2*n+1)));
    
    

   







Plotting; %plot all returned values
