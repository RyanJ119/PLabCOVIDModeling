function [cost] = theModel(u)

Inputs; %bring in all chosen input values

%% Call the solver i times
for c = 1:runSolver

    tRange = daysUpdate*(runSolver-1):1:daysUpdate*runSolver;   %% number of days to run before breaking out of solver to mutate infected

    %     if  length(R)>300
    %
    %         Ro=zeros(n);
    %         beta=Ro.*gamma; %%%%Use this to change replication rate mid run
    %
    %     end

     ustep = floor(length(S)/daysBetweenGovtUpdates)+1;


    [tSol,YSol] = ode45(@(t,Y) ODESolver(t,Y, n, betaRate', gammaRate', u(ustep),betaHat, sigma',eta', d), tRange, Yo);

    %%%% concatinating matrices double counts last previous entry/first new
    %%%% entry. Delete one of them here, then concatonate old solution with new
    %%%% solution

    if length(S)>0
        S(end)=[];
    end

    S = vertcat(S, YSol(:,1));

    if length(I)>0
        I(end, :) = [] ;
    end

    I = vertcat(I,  YSol( :, 2:(n+1) ));

    if length(R)>0
        R(end, :) = [] ;
    end

    R = vertcat(R, YSol( :, (n+2) : 2*n+1 ));

    if length(Sr)>0
        Sr (end, :) = [] ;
    end
    Sr = vertcat(Sr, YSol(:,2*n+2 : 3*n+1));


        if length(H)>0
        H (end, :) = [] ;
    end
    H = vertcat(H, YSol(:,3*n+2 : 4*n+1));



    if length(D)>0
        D(end)=[];
    end

    D = vertcat(D, YSol(:,4*n+2));




   updatingStep; %virus mutating step


end


uTimeHorizon = repelem(u, daysBetweenGovtUpdates); % define the u vector of length(S) size for cost function
uTimeHorizon = horzcat(1, uTimeHorizon );

sumHosp =sum(max(diff(sum(H,2)),0));

deaths = D(end);


 cost =  costFunction(S, sumHosp,deaths, uTimeHorizon);



end
