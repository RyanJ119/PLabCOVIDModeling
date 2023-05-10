function [cost] = costfunction(currentSusceptible,hosppitalizations, deaths )
c1 = 70
c2 = 2700
c3 = 1500000
cost1 = c1*currentSusceptible(1-u)
cost2 = c2*hosppitalizations
cost3 = c3*deaths
return cost1+cost2+cost3
end
