
tRange = 0:1:length(S)-1;
tSol = tRange;                  %Define time range  for plotting
uRange =  daysBetweenGovtUpdates:daysBetweenGovtUpdates:totalDays;
plotU = zeros(1, 300);
for i = 1:length(u)
    plotU((i-1)*daysBetweenGovtUpdates+1: daysBetweenGovtUpdates*i) = u(i);

end

sumH = sum(H,2);
sumSr = sum(Sr,2);
sumR = sum(R,2);

sumItotal = 0; %set the total infected to zero before summing
for i = 1:n
    %     plot(tSol,I(:,i), 'yellow');  Plotting all mutations of variants if desired
    sumItotal = sumItotal+I(:,i);  %sum Infected for total infected number
end

sumI = zeros(n/vas, totalDays+1)';   %set up a matrix to conatenate our infected populations into specific variants


for j = 1:vas
    for i = 1:(n/vas)

        sumI(:,j) = sumI(:,j)+I(:,(j-1)*(n/vas)+i);  %concatenate n infected populations into vas categories

    end

end


figure('name','dynamicsOfMDECode'); %Plotting the dynamics altogether

h(1) = plot(tSol,S, 'DisplayName', 'susceptible');
hold on;
for i = 1:vas
    h(i+1) =     plot(tSol, sumI(:,i),'DisplayName', ['Infection ' num2str(i)]); %plot new infected;
end

h(vas+2) = plot(tSol, sumItotal, 'DisplayName', 'Total Infected ') ;

h(vas+3) = plot(tSol,sumR, 'DisplayName', 'Recovered ');
h(vas+4) = plot(tSol,sumH, 'DisplayName', 'Hospitalized ');
h(vas+5) = plot(tSol,sumSr, 'DisplayName', 'Susceptible but recovered ');
legend(h, 'FontSize', 18)
xlabel("Days", 'FontSize', 18)
ylabel("Number of Individuals", 'FontSize', 18)

%h = [];

figure; %Plotting u

 plot(tSol(1:length(plotU)),1-plotU);
%title('Percent Locked Down',  'FontSize', 16)
xlabel('Days', 'FontSize', 20);
ylabel('Percent Locked Down', 'FontSize', 20);
ax = gca;
ax.FontSize = 20;
figure('name','input values'); %Plotting R Beta and Gamma

subplot(3,3,1);
plot(mu,gammaRate)
title('\gamma(\alpha)')
subplot(3,3,2);
plot(mu,Ro)
title('Replication Rate')
subplot(3,3,3);
plot(mu,betaRate)
title('\beta(\alpha)')
subplot(3,3,4);
plot(mu,phi(mu))
title('\Phi(\alpha)')
subplot(3,3,5);
plot(mu,eta)
title('\eta(\alpha)')
figure;



%I Plots
 ss = size(I);
 [x,y] = ndgrid(1:ss(2),1:ss(1));

surf(y,x,I.');
colormap turbo
c = colorbar;
%caxis([0 1e4])
shading interp
xlabel('Days')
ylabel('Mutations')
zlabel('Infected')



figure;
 ss = size(probdist);
 [x,y] = ndgrid(1:ss(2),1:ss(1));

 probmat = repelem(probdist,2, 1);

 imagesc(probmat');
myColorMap = jet(256);
myColorMap(1,:) = 1;
colormap(myColorMap);
colorbar()
caxis([0 .0005])
axis square
h = gca;
% h.XTick = 1:4;
% h.YTick = 1:4;
title 'Probability Distribution';





%R Plots

ss = size(R);
 [x,y] = ndgrid(1:ss(2),1:ss(1));

surf(y,x,R.');
colormap turbo
c = colorbar;
%caxis([0 1e4])
shading interp
xlabel('Days')
ylabel('Mutations')
zlabel('Recovered')




ss = size(Sr);
 [x,y] = ndgrid(1:ss(2),1:ss(1));

surf(y,x,Sr.');
colormap turbo
c = colorbar;
%caxis([0 1e4])
shading interp
xlabel('Days')
ylabel('Mutations')
zlabel('Susceptible Recovered')



%surf(2*y,x,probdist.');
% colormap turbo
% c = colorbar;
%caxis([0 .1])
% shading interp
% xlabel('Days')
% ylabel('Mutations')
% zlabel('Percent of total infections')
