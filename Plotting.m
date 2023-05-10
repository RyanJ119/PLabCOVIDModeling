

tRange = 0:1:length(R)-1;
tSol = tRange;                  %Define time range  for plotting




sumItotal = 0; %set the total infected to zero before summing
for i = 1:n
    %     plot(tSol,I(:,i), 'yellow');  Plotting all mutations of variants if desired
    sumItotal = sumItotal+I(:,i);  %sum Infected for total infected number
end

sumI = zeros(n/aas, totalDays+1)';   %set up a matrix to conatenate our infected populations into specific variants









figure('name','input values'); %Plotting R Beta and Gamma

subplot(2,2,1);
plot(mu,gamma)
title('Gamma')
subplot(2,2,2);
plot(mu,Ro*n)
title('Replication Rate')
subplot(2,2,3);
plot(mu,beta*n)
title('Beta')


figure;

subplot(2,1,1);
 ss = size(I);
 [x,y] = ndgrid(1:ss(2),1:ss(1));

surf(y,x,I.');
colormap turbo
c = colorbar;
%caxis([0 1e4])
shading interp
xlabel('Days')
ylabel('Age')
zlabel('Infected')


figure;

subplot(2,1,1);
 ss = size(S);
 [x,y] = ndgrid(1:ss(2),1:ss(1));

surf(y,x,S.');
colormap turbo
c = colorbar;
%caxis([0 1e4])
shading interp
xlabel('Days')
ylabel('Age')
zlabel('Susceptible')











