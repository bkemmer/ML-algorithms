%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perceptron com fun��o softmax 
% X      - matriz de entrada
%yd      - matriz de saida desejada
% nepmax - numero de epocas m�xima
% w      - matriz de pesos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w=perceptron_soft(X,yd,nepmax) 
[N,m]=size(X);
[~,nc]=size(yd);
w=rand(nc,m);
y=calc_saida_soft(X,w,nc);
erro = y-yd;
EQM=1/N*sum(sum(erro.*erro));
nep=0;
alfa=0.1;
vetEQM=[];
vetEQM=[vetEQM;EQM];
while EQM>1e-5 & nep<nepmax
    nep = nep+1;
    dJdw=calc_grad_soft(X,yd,w,nc,m,N);
    w =w - alfa*dJdw;
    y=calc_saida_soft(X,w,nc);
    erro = y-yd;
    EQM=1/N*sum(sum(erro.*erro));
    vetEQM=[vetEQM;EQM];
end
plot(0:nep,vetEQM)
end

function y=calc_saida_soft(X,w,nc)
yin = X*w';
y = exp(yin)./(sum(exp(yin),2)*ones(1,nc));
end


function dJdw=calc_grad_soft(X,yd,w,nc,m,N)
yin = X*w';
y = exp(yin)./(sum(exp(yin),2)*ones(1,nc));
erro = y-yd;
dJdw=zeros(nc,m);
for n=1:N,
    for i=1:nc
        for j=1:m
            for k=1:nc
                if i==k
                    dJdw(i,j) = dJdw(i,j) + erro(n,k)*(1-y(n,k))*y(n,i)*X(n,j);
                else
                    dJdw(i,j) = dJdw(i,j) + erro(n,k)*(0-y(n,k))*y(n,i)*X(n,j);
                end
            end
        end
    end
end
dJdw = 1/N*dJdw;
end


X = [0,0;0,1;1,0;1,1]
yd = [1, 0; 1, 1; 1, 1; 0, 1]

w=perceptron_soft(X, yd, 1000)
print(w)