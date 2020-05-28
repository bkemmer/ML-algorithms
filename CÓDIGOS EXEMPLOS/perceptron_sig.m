%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perceptron com fun��o sigmoid 
% X      - matriz de entrada
%yd      - matriz de saida desejada
% nepmax - numero de epocas m�xima
% w      - matriz de pesos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function w=perceptron_sig(X,yd,nepmax) 
[N,m]=size(X);
[~,nc]=size(yd);
w=rands(nc,m);
y=calc_saida_sig(X,w);
erro = y-yd;
EQM=1/N*sum(sum(erro.*erro));
nep=0;
alfa=0.5;
vetEQM=[];
vetEQM=[vetEQM;EQM];
while EQM>1e-5 & nep<nepmax
    nep = nep+1;
    dJdw=calc_grad_sig(X,yd,w,nc,N);
    w =w - alfa*dJdw;
    y=calc_saida_sig(X,w);
    erro = y-yd;
    EQM=1/N*sum(sum(erro.*erro));
    vetEQM=[vetEQM;EQM];                            
end
plot(0:nep,vetEQM)
end

function y=calc_saida_sig(X,w)
yin = X*w';
y = 1./(1+exp(-yin));
end


function dJdw=calc_grad_sig(X,yd,w,nc,N)
yin = X*w';
y = 1./(1+exp(-yin));
erro = y-yd;
dJdw = 1/N*(erro.*(1-y).*y)'*X;
end

