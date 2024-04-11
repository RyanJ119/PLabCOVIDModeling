function [cost] = costfunction(currentSusceptible,hospitalizations, deaths, u )
c1 = 70;
c2 = 2700;
c3 = 1500000;
cost1 = c1*dot(currentSusceptible, (1-u)); %cost of social distancing
cost2 = c2*hospitalizations; %cost of hospitalization
cost3 = c3*deaths; %cost of death

cost =  cost1+cost2+cost3;
end
