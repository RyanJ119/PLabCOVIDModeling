flip = I';              %Turn I in order to do mutation step
Iend = flip(:, end);


probdist(c,:) = Iend'./sum(Iend);

    F = zeros(n);

    for k = 1:n
        F(k) = sum(probdist(c,1:k));

        % 0 0 0 0 .2 0 0 0 .54 .26
        if F(k)>0
        for j = 1:ceil(probdist(c,k)/delm)

            if probdist(c, k) >=delm



                if      j ~= ceil(probdist(c, k)/delm) && j>1
                    F2(j) = F2(j-1)+delm;
                    pieces(j) = delm;
                elseif      j == 1
                    F2(j) = delm;
                    pieces(j) = delm;
                elseif j == ceil(probdist(c, k)/delm)           %Split mass at a point into pieces sized delm
                    F2(j) = probdist(c, k);
                    pieces(j) =probdist(c, k)-F2(j-1) ;
                end
            end
            if probdist(c, k) <=delm
                F2(j) = probdist(c, k);
                pieces(j) = probdist(c, k);
            end


            %vel = zeros(1,j);
            if k > 1
                vel(j) = floor((1/delv)*phi(F(k-1)+F2(j))+0.000001);

            elseif k == 1
                vel(j) = 0;
            end

            e1 = zeros(1, n);
            e2 = zeros(1, n);
            e1(k) = 1;

            if (k+vel(j))<=n && (k+vel(j)>=1)

                e2(k+vel(j)) = 1;

                %  e2
            elseif (k+vel(j))>n
                e2(n) = 1;

            elseif (k+vel(j))<1
                e2(1) = 1;

            end
            probdist(c+1,:) =  probdist(c+1,:)+e2*pieces(j);

            %         if  e2 ~=zeros(1,n)
            %             e2
            %         end
        end
        end
    end

    for w = 1:runSolver+1
        for q = 1:n
            if probdist(runSolver+1, n) < .0000005
                probdist(runSolver+1, n) = 0;
            end
        end
    end



    Iend = probdist(c+1,:).*sum(Iend);





I(end, :) = Iend;

Yo = [S(end); I(end, :)' ;R(end, :)';Sr(end, :)'; H(end, :)' ; D(end)];
