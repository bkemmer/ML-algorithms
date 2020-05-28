function hebb()
clc
x=[1 1 0 0;1 0 1 0];
t=[1 -1 -1 -1];
%Primeiro Passo
wnew=[0 0];
wold=[0 0];
bnew=0;
bold=0;
alfa=0.1;
theta=0.01;

%Segundo Passo
cont=1; 
while cont==1
    
    for i=1:length(t)
        
        y_in=bnew+wnew*x(:,i);
        if y_in>=theta
            y=1;
        else
            if y_in>=-theta
                y=0;
            else
                y=-1;
            end
        end
        if y~= t(i)
            wnew=wnew+alfa*t(i)*x(:,i)';
            bnew=bnew+alfa*t(i);
        end
    end
     
    if norm(wnew-wold)==0 & abs(bnew-bold)==0
        cont=0;
    end
    wold=wnew;
    bold=bnew;
    x1=-1.5:2.5;
    x2=-wnew(1)/wnew(2)*x1-(bnew-theta)/wnew(2);

    figure(1)
    clf
    plot(x(1,2:end),x(2,2:end),'bo','linewidth',3)
    grid on
    hold on
    plot(x(1,1),x(2,1),'ro','linewidth',3)
    plot(x1,x2,'r','linewidth',3)   
    x2=-wnew(1)/wnew(2)*x1-(bnew+theta)/wnew(2);
    plot(x1,x2,'b','linewidth',3)
    
    pause
end
