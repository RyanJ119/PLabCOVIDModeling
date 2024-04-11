
Inputs; %bring in all chosen input values
%%%%%% Update every 30 days but starting on day 90
%u = [1,1,1,0.522073932757273,0.516454769139200,0.512303238472779,0.453175556093288,0.284124808599377,0.502844956155776,0.0696587524596329,0.997689242462265,0.999983442461855];


%%%%%% Update every 30 days but starting on day 150
%u =[1,1,1,1,1,0.000264584511507792,1.15846711667817e-05,0.446162455787719,0.463119434002227,0.116160766435834,0.926159836883534,0.999999991580023];


%%%%%% Update every 60 days starting on day 60
%u = [1,0.805644856382037,0.441916303338130,0.452383434254045,0.222325235022525,0.999999999995839];


u = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
%%%%%% Update every 30 days starting on day 60
% u = [1,1,0.934933343861075,0.9,0.9,0.9,0.466623805758369,0.406166110519103,0.400027523795879,0.174310895863716,0.899043873767161,0.999999592836230];
%u = [1,1,1,1,3.11578784993042e-08,0.411221075570975,0.323750355635756,0.00331690557671753,6.20300460346991e-05,0.749511186755973,0.999999999710127,0.999943053416217]
%u = x
%u = [1,1,0.999986478306374,0.453461252655134,0.399720639396797,0.352803746910239,0.269106988028164,0.179141401582316,0.282431396193834,0.496413583710368,0.999990653969968,0.999981948043570]




%u = ones(1, length(u));


cost = 0;
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

sumHosp =sum(abs(diff(H)));

deaths = D(end)


 cost =  costFunction(S, sumHosp,deaths, uTimeHorizon);



Plotting; %plot all returned values
