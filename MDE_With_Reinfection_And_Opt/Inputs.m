%   Model for progression of disease with variants

%input parameters: vas (number of variants acting on the system), n (mutations of variants), R0(replications rate), Beta(infection
%rate), Gamma(recovery rate), markov chain (mutation factor),

vas = 1; %number of variants affecting the system: variants associated with specific data used to build R function
n = 500; %number of different mutations of variants n>1 (use multiples of vas)
mu = linspace(0,1,n) ; %x range of distributions: taking 0-1 and placing n evenly distributed points
initI = zeros(1, n); %initialize the infected populations with zeroes
initSr = zeros(1, n);
initR = zeros(1, n);
initH = zeros(1, n);
% for i = 1:n
initD=0;
% initI(i) = 1; %set middle element to one infected
% end
initI(n/2) = 1;

%  for i = 1:vas
%  initI((i/vas)*n) = 1;     %update the initial infected populations using values or functions
%  end

%initI = variants/100+1;% this will need to be fixed to be a function matching the four variants
initialS = 8.882e6;

Yo = [initialS; initI' ;initR'; initSr'; initH'; initD];%% Initial S, I1, I2,... In, R, H, D


S = []; %susceptible populations
I = [];% infected pop
R = [];%recovered pop
H = []; %hospitalized pop
Sr = [];
D = []; %hospitalized pop
Iend=[];
% T = [0.9 0.1 0.00 0.00; %A
%     0.02 0.9 0.03 0.05; %B
%     0.02 0.03 0.9 0.05; %G
%     0.02 0.03 0.05 .9]; %D

Ro=-sin(9*mu+21.9)+2.5 ;

daysBetweenGovtUpdates = 30;

%Ro=(mu-.5).^2+1.75 ; % replication rate function
gammaRate = (mu+1)./(mu+1) -13/14; % recovery rate (here it is just 1/14
betaRate=Ro.*gammaRate; %find betaRate for all infected populations
eta= (mu+1)./(mu+1)-1 + .01; %hospitalization rate

%sigma= 0; %hospitalization rate


sigma =( (mu+1)./(mu+1) -119/120); % 1/60 
%sigma = .066*(mu-.5).^2+(1/60)


daysUpdate = 2; % Number of days between mutations (swapping between infected groups)
totalDays = 900; %total days of program
runSolver = totalDays/daysUpdate;  % number of times to run the solver
probdist = zeros(runSolver+1,n);

betaHat = zeros(n, n);
 for i= 1:length(betaRate)
     betaHat(i,:) = betaRate;
     for j= 1: 1:length(betaRate)
         if  i+j<length(betaRate) && abs(i-j)<= 10
             betaHat(i,j) = betaHat(i,j)*(i-j)*abs(i-j)*.0001;
         end
    end
 end


d = .19;
%u = ones(1,runSolver);

%u = .99; %1 = no lockdown 0 = full lockdown
%% Mutating constants 

    m = 10;
    v = .5*m;       %Speed at which mutation will travel 
    delv = 1/v;
    delm = 1/m;     %Size of one piece of infected to be mutated
    
