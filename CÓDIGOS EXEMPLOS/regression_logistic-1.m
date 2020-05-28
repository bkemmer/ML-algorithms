%************************************************************************
% Disciplina Aprendizado de Máquina - 15/04/2020
% Autor Clodoaldo A M Lima
% N - Número de instâncias
% m - Número de atributos
% nc - Número de classes
% X - matriz (Nxm) de dados de entrada
% Y - matriz (Nxnc) de dados de saida
% ***********************************************************************

function w=regression_logistic(X,Y)
[N,m]=size(X);
[N,nc]=size(Y);

if nc == 1 % codificação {1,-1}
    disp('Problema Binário')
    w=logistica_binario(X,Y,N,m);
else %codificação 1-of-nc 
    disp('Problema Com Múltiplas Classes')
    w=logistica_multclass(X,Y,N,m,nc);
end

end


function w=logistica_multclass(X,Y,N,m,nc)
w = rands(nc,m); %inicializa a matriz de pesos
Yr=calc_saida_multclass(X,Y,w,nc);
E=fobjetiva_multclass(X,Y,w,N,nc);
it=0;
itmax = 2000;
alfa =0.1;
E_ant = Inf;
E_new = E;
vet_E =[];
vet_E = [vet_E;E];
while it<itmax & abs(E_new-E_ant)>1e-3
    it = it +1;
    disp(sprintf('Função Objetiva = %2.2f, iteração %d',E,it))
    [dEdW,dE2dW2] = derivada_multclass(X,Y,w,N,m,nc);
    dE2dW2=check_cond(dE2dW2,m,nc);
    dir = -inv(dE2dW2)*reshape(dEdW',nc*m,1);
    %dir = -reshape(dEdW',nc*m,1);
    alfa = calc_alfa_multiclass(X,Y,w,dir,N,m,nc);
    dir = reshape(dir,m,nc)';
    w =w + alfa*dir;
    E=fobjetiva_multclass(X,Y,w,N,nc);
    E_ant = E_new;
    E_new = E;
    vet_E = [vet_E;E];
end
plot(0:it,vet_E)
end

function H=check_cond(H,m,nc)
[V,D] = eigs(H);
D = min(diag(D));
while D<1e-8
    H = H + 1e-8*eye(nc*m); 
    [V,D] = eigs(H);
    D = min(diag(D));
end
end

function alfa =calc_alfa_multiclass(X,Y,w,dir,N,m,nc)
epsilon = 1e-3;
hlmin = 1e-3;
alfa_l = 0;
alfa_u = rand();
dir1 = reshape(dir,m,nc)';
wn = w + alfa_u*dir1;
[g,~] = derivada_multclass(X,Y,wn,N,m,nc);
g = reshape(g',nc*m,1);
hl = g'*dir;
while hl<0
    alfa_u = 2*alfa_u;
    wn = w + alfa_u*dir1;
    [g,~] = derivada_multclass(X,Y,wn,N,m,nc);
    g = reshape(g',nc*m,1);
    hl = g'*dir;
end
alfa_m = (alfa_l+alfa_u)/2;
itmax = ceil(log ((alfa_u-alfa_l)/epsilon));
it = 0;
while abs(hl)>hlmin & it<itmax
    it = it + 1;
    wn = w + alfa_m*dir1;
    [g,~] = derivada_multclass(X,Y,wn,N,m,nc);
    g = reshape(g',nc*m,1);
    hl = g'*dir;
    if hl>0
        alfa_u = alfa_m;
    elseif hl<0
        alfa_l = alfa_m;
    else
        break;
    end
    alfa_m = (alfa_l+alfa_u)/2;   
end
alfa = alfa_m;
end

function E=fobjetiva_multclass(X,Y,w,N,nc)
Yr = X*w';
Yr = exp(Yr)./(sum(exp(Yr),2)*ones(1,nc));
E = -sum(sum(Y.*log(Yr+eps)));
end

function Yr=calc_saida_multclass(X,Y,w,nc)
Yr = X*w';
Yr = exp(Yr)./(sum(exp(Yr),2)*ones(1,nc));
end

function [dEdW,dE2dW2] = derivada_multclass(X,Y,w,N,m,nc)
dE2dW2 = zeros(m*nc,m*nc);
Yr = X*w';
Yr = exp(Yr)./(sum(exp(Yr),2)*ones(1,nc));

erro = Yr - Y;
dEdW = erro'*X;


for n=1:N,
      for k=1:nc,
          for j=1:nc,
              if k==j
                  dE2dW2_aux = Yr(n,k)*(1-Yr(n,j))*X(n,:)'*X(n,:);
              else
                  dE2dW2_aux = Yr(n,k)*(0-Yr(n,j))*X(n,:)'*X(n,:);
              end
              dE2dW2(1+(k-1)*m:k*m,1+(j-1)*m:j*m)=dE2dW2(1+(k-1)*m:k*m,1+(j-1)*m:j*m)+dE2dW2_aux;
             %dE2dW2
          %pause
          end
          
      end
end

end

function w=logistica_binario(X,Y,N,m)
w = rands(1,m);
Yr=calc_saida_binario(X,Y,w);
E=fobjetiva_binario(X,Y,w,N);
it=0;
itmax = 2000;
%alfa =0.1;
E_ant = Inf;
E_new = E;
vet_E =[];
vet_E = [vet_E;E];
while it<itmax & abs(E_new-E_ant)>1e-8
    it = it +1;
    disp(sprintf('Função Objetiva = %2.2f, iteração %d',E,it))
    dEdW = derivada_binario(X,Y,w,N,m);
    dir = -dEdW;
    alfa =calc_alfa_binario(X,Y,w,dir,N,m);
    w =w - alfa*dEdW;
    E=fobjetiva_binario(X,Y,w,N);
    Yr=calc_saida_binario(X,Y,w);
    E_ant = E_new;
    E_new = E;
    vet_E = [vet_E;E];
   
end
plot(0:it,vet_E)
end

function E=fobjetiva_binario(X,Y,w,N)
E = 1/N*sum(log2(1+exp(-Y.*(X*w'))));

end

function Yr=calc_saida_binario(X,Y,w)
Yin = X*w';
Yr = 1./(1+exp(-Yin));
end

function dEdW = derivada_binario(X,Y,w,N,m)
dEdW=zeros(1,m);
for i=1:N,
    dEdW = dEdW + Y(i,1)*X(i,:)/(1+exp(Y(i,1)*X(i,:)*w'));
end
dEdW = -dEdW/N;
end

function alfa =calc_alfa_binario(X,Y,w,dir,N,m)
epsilon = 1e-3;
hlmin = 1e-3;
alfa_l = 0;
alfa_u = rand();
wn = w + alfa_u*dir;
g = derivada_binario(X,Y,wn,N,m);
hl = g'*dir;
while hl<0
    alfa_u = 2*alfa_u;
    wn = w + alfa_u*dir;
    g = derivada_binario(X,Y,wn,N,m);
    hl = g'*dir;
end
alfa_m = (alfa_l+alfa_u)/2;
itmax = ceil(log ((alfa_u-alfa_l)/epsilon));
it = 0;
while abs(hl)>hlmin & it<itmax
    it = it + 1;
    wn = w + alfa_m*dir;
    g = derivada_binario(X,Y,wn,N,m);
    hl = g'*dir;
    if hl>0
        alfa_u = alfa_m;
    elseif hl<0
        alfa_l = alfa_m;
    else
        break;
    end
    alfa_m = (alfa_l+alfa_u)/2;   
end
alfa = alfa_m;
end