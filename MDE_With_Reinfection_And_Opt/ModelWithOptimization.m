clear all;
Inputs; 

% %theModel(.5)

% %x = fminsearch(theModel,initialGuess)
 %uOpt  = fminbnd(@(uOpt)theModel(uOpt),0, 1)
% %u = fminbnd(@costFunction,0,1)
% %u = fminsearch(@Model,[0,1])
% goal = 0;
% weight = ones(1,runSolver);
% 
% x = fgoalattain(phi,1,0,1 )
%A = ones(1,runSolver);
%b = 400;
%updates = 1;









updates = ceil(totalDays/daysBetweenGovtUpdates);
 initialZero = zeros(1,updates);
 x0 = initialZero+.9;
lb = zeros(1,updates);
%lb(1:4) = 1;
ub = ones(1,updates);
A = [];
b = [];
Aeq = [];
beq = [];
%x0 = [1,1,1,1,3.11578784993042e-08,0.411221075570975,0.323750355635756,0.00331690557671753,6.20300460346991e-05,0.749511186755973,0.999999999710127,0.999943053416217]

%problem = createOptimProblem('fmincon', 'x0',x0,'objective', theModel, 'lb', lb, 'ub', ub  )
opts = optimset('MaxIter',40,'MaxFunEvals',3000,'Display','iter', 'MaxTime', 3600);%'Algorithm','sqp',
problem = createOptimProblem('fmincon', 'x0',x0,'objective', @(uOpt)theModel(uOpt), 'lb', lb, 'ub', ub , 'options',opts  )

ms = MultiStart('UseParallel', true)

%%%% Use gs to find a global minimum8
%gs = GlobalSearch;
%x = run(gs,problem)
%gs = GlobalSearch(ms, 'MaxTime', 3600)
%x = run(gs,problem)


%%%%%%%%%% Local minimum, should be used given a good x0
x = fmincon(@(uOpt)theModel(uOpt),x0,A,b, Aeq,beq , lb, ub,[], opts) 


%x = run(ms,problem,20)



