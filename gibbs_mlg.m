function [W,beta,sigma2,alphabeta,alphaeta,kappabeta,kappaeta,theta]=gibbs_usmlg_70417(B,data,X,y_m,locs)
n = size(data,1);
p = size(X,2);
W = zeros(n,B);
beta = zeros(p,B);
sigma2 =0.008*ones(1,B);
alphabeta = ones(1,B);
kappabeta = ones(1,B);
alphaeta = 1*ones(1,B);
kappaeta = 1*ones(1,B);
theta =ones(1,B);
Distmatrix = squareform(pdist(locs));
V=log(data)-log(y_m);

for b = 2:B

    %sample W
    
    Sig = exp(-theta(b-1)*Distmatrix);
    cov_w=sqrtm(sigma2(b-1).*Sig);
%     d=1/(1+max(abs(ones(1,n)*cov_w)));
    H_w(1:n,1:n)=eye(n);
    H_w(n+1:2*n,1:n)=inv(cov_w);
%     alpha_w(1:n,1)=ones(n,1)+d.*ones(n,1);
%     alpha_w(n+1:2*n,1)=ones(n,1)-d.*(cov_w'*ones(n,1));
    alpha_w(1:n,1)=ones(n,1);
    alpha_w(n+1:2*n,1)=alphaeta(b-1)*ones(n,1);
    kappa_w(1:n,1)=V.*exp(X*beta(:,b-1));
    kappa_w(n+1:2*n,1)=kappaeta(b-1)*ones(n,1);
    W(:,b)=cmlg(H_w,alpha_w,kappa_w); 
   
    %sample beta
    H_beta(1:n,1:p)=X;
    H_beta(n+1:n+p,1:p)=inv(sqrtm(100*eye(p)));
%     d=1/(1+max(abs(ones(1,p)*sqrtm(100.*eye(p)))));
%     alpha_beta(1:n,1)=ones(n,1)+d.*ones(n,1);
%     alpha_beta(n+1:n+p,1)=ones(p,1)-d.*(10*eye(p)*ones(p,1));
    alpha_beta(1:n,1)=ones(n,1);
    alpha_beta(n+1:n+p,1)=alphabeta(b-1)*ones(p,1);
    kappa_beta(1:n,1)=V.*exp(W(:,b));
    kappa_beta(n+1:n+p,1)=kappabeta(b-1)*ones(p,1);
    beta(:,b)=cmlg(H_beta,alpha_beta,kappa_beta);
    
    
   
%     %sample sigma2
%   sigma2t= @(r) sigma2fullcond3_62017(W(:,b),r,alphaeta(b-1)*ones(n,1),kappaeta(b-1)*ones(n,1),theta(b-1),Distmatrix);
%   sigma2(b)=slicesample(sigma2(b-1),1,'logpdf',sigma2t,'thin',10);
    %update theta
    
  
%     %sample alphaeta
     sigma2t= @(r) alphakappa_W_fc(W(:,b),sigma2(b),theta(b-1),r*ones(n,1),kappaeta(b-1)*ones(n,1),Distmatrix,1);
     alphaeta(b)=slicesample(alphaeta(b-1),1,'logpdf',sigma2t);
   %alphaeta(b)=mhsample(alphaeta(b-1),1,'logpdf',sigma2t,'proprnd',@(r) normrnd(r,0.1),'symmetric',1);
%     %sample alphaeta
     sigma2t= @(r) alphakappa_W_fc(W(:,b),sigma2(b),theta(b-1),alphaeta(b)*ones(n,1),r*ones(n,1),Distmatrix,0);
     kappaeta(b)=slicesample(kappaeta(b-1),1,'logpdf',sigma2t);

     sigma2t =@(r) alphakappa_beta_fc(beta(:,b),r*ones(p,1),kappabeta(b-1)*ones(p,1),1);
     alphabeta(b)=slicesample(alphabeta(b-1),1,'logpdf',sigma2t);
     
     sigma2t =@(r) alphakappa_beta_fc(beta(:,b),alphabeta(b)*ones(p,1),r*ones(p,1),0);
     kappabeta(b)=slicesample(kappabeta(b-1),1,'logpdf',sigma2t);
     
   %kappaeta(b)=mhsample(kappaeta(b-1),1,'logpdf',sigma2t,'proprnd',@(r) normrnd(r,0.1),'symmetric',1);
    %update theta
    
    thetat= @(r) thetafullcond3_62017(W(:,b),sigma2(b),alphaeta(b)*ones(n,1),kappaeta(b)*ones(n,1),r,Distmatrix);
%    theta(b)=slicesample(theta(b-1),1,'logpdf',thetat);
    %theta(b)=mhsample(theta(b-1),1,'logpdf',thetat,'proprnd',@(r) gamrnd(1,1),'symmetric',1);
    temp=normrnd(theta(b-1),0.5);
    if temp>0
    prop=exp(thetat(temp)-thetat(theta(b-1)));
    if prop>unifrnd(0,1)
        theta(b)=temp;
    else
        theta(b)=theta(b-1);
    end
    else
        theta(b)=theta(b-1);
    end
    
    
    disp(b)
end

end
