%% Developer: Partha Majumder
%%
clc
clear all
close all
format long g

    
%% Get Details of objective function
%% Unimodal function
[LB,UB,D,fobj] = Get_Functions_details('Sphere1_F39'); % Fg=0;  D=D
% [LB,UB,D,fobj] = Get_Functions_details('Rosenbrock1_F31'); % Fg=0; D=D
% [LB,UB,D,fobj] = Get_Functions_details('Griewank_F15');  % Fg=0; D=D

%% Multimodal function
% [LB,UB,D,fobj] = Get_Functions_details('Ackley_F1');  % Fg=0; D=D
% [LB,UB,D,fobj] = Get_Functions_details('Rastrigin1_F30'); % Fg=0; D=D


% [LB,UB,D,fobj] = Get_Functions_details('HimmelblaufcnF44'); % Fg=0; D=2
% [LB,UB,D,fobj] = Get_Functions_details('Styblinski_Tang_F40');  % Fg=-39.16599*D 
 %[LB,UB,D,fobj] = Get_Functions_details('Shubert4fcn45');  % Fg=-386.12
% [LB,UB,D,fobj] = Get_Functions_details('alpinen2fcn46');  % Fg=2.808^n

%% Other Functions
% [LB,UB,D,fobj] = Get_Functions_details('Six_Hump_Camel_F38');  % Fg=-1.031628453486

%% DE PARAMETERS (1)
beta_min=0.2;     % Lower Bound of Scaling Factor
beta_max=0.8;     % Upper Bound of Scaling Factor
pCR=0.2;          % Crossover Probability
VarSize=[1 D];
%% DE PARAMETERS (1)

itmax=400; % Maximum numbef of iterations
N=80;

if length(LB)==1
for kk=1:1:D
    lb(1:N,kk)=LB; 
    ub(1:N,kk)=UB;
end
end

if length(LB)~=1
for kk=1:1:D
    lb(1:N,kk)=LB(kk); 
    ub(1:N,kk)=UB(kk);    
end
end


Alpha_pos=zeros(1,D);
Alpha_score=inf;    % F_alp
Beta_pos=zeros(1,D);  
Beta_score=inf;     % F_bet
Delta_pos=zeros(1,D);  
Delta_score=inf;    % F_del

x=lb+(ub-lb).*rand(N,D);
Fgbest_vect=zeros(1,itmax);

%% DE initial computation
for ii=1:1:N
    F_DE(ii) = fobj(x(ii,:),D);
end
[F_g_bestDE,pp]=min(F_DE);
g_best_DE=x(pp,:);
%% DE initial computation


for it=1:1:itmax
    a=2*(1-it/itmax);
    for ii=1:1:N 
        %% Elite Opposition Based Learning
        x_O=lb(ii,:)+ub(ii,:)-x(ii,:);  % Opposite number
        F_O=fobj(x_O,D);
        %% Elite Opposition Based Learning      
        
        F= fobj(x(ii,:),D);
        if F_O<F
            %disp('improvement from opposition based learning:')
            F=F_O;
            x(ii,:)=x_O;
        end
        if F<Alpha_score
            Alpha_score=F;
            Alpha_pos=x(ii,:);
        end
        
        if F>Alpha_score && F<Beta_score
            Beta_score=F;
            Beta_pos=x(ii,:);
        end
        
        if F>Alpha_score && F>Beta_score && F<Delta_score
            Delta_score=F;
            Delta_pos=x(ii,:);
        end    
    end
    for ii=1:1:N
        for jj=1:1:D
            r1=rand();
            r2=rand();
            A1=(2*r1-1)*a;
            C1=2*r2;
            D_alpha=abs(C1*Alpha_pos(jj)-x(ii,jj));
            X1=Alpha_pos(jj)-A1*D_alpha;
            
            r1=rand();
            r2=rand();
            A2=(2*r1-1)*a;
            C2=2*r2;
            D_beta=abs(C2*Beta_pos(jj)-x(ii,jj));
            X2=Beta_pos(jj)-A2*D_beta;
            
            
            r1=rand();
            r2=rand();
            A3=(2*r1-1)*a;
            C3=2*r2;
            D_delta=abs(C3*Delta_pos(jj)-x(ii,jj));
            X3=Delta_pos(jj)-A3*D_delta;
            x(ii,jj)=(X1+X2+X3)/3;           
            
        end
    end
    for kk=1:1:D
        x(:,kk)=min(x(:,kk),ub(:,kk));
        x(:,kk)=max(x(:,kk),lb(:,kk));
    end
    
    %% DE
    for ii=1:1:N
        x_vecT=x(ii,:);
        vect=randperm(N);
        vect(vect==ii)=[];
        ri1=vect(1);
        ri2=vect(2);
        ri3=vect(3);
        
        %% Mutation
        beta=unifrnd(beta_min,beta_max,VarSize);
        y=x(ri1,:)+beta.*(x(ri2,:)-x(ri3,:));
        y = max(y, lb(1,:));
        y = min(y, ub(1,:));
        %% Crossover
        z_c=zeros(size(x_vecT));       %Crossover vector
        J0=randi([1 length(x_vecT)]);
        for jj=1:length(x_vecT)
            if jj==J0 || rand<=pCR
                z(jj)=y(jj);
            else
                z(jj)=x_vecT(jj);
            end
        end
        
        Fz= fobj(z,D);
        if  Fz<F_DE(ii)
            x(ii,:)=z;
            F_DE(ii)=Fz;
            if F_DE(ii)<F_g_bestDE
                F_g_bestDE=F_DE(ii);
                g_best_DE=x(ii,:);
            end
        end
    end
    
    for kk=1:1:D
        x(:,kk)=min(x(:,kk),ub(:,kk));
        x(:,kk)=max(x(:,kk),lb(:,kk));
    end
    
    %% DE
    
    %% Elimination Mechanism
    percent=0.2;
    [value, INDEX] = sort(F_DE,'ascend');
    for iiu=1:1:length(INDEX)
        x_sorted(iiu,:)=x(INDEX(iiu),:);
    end
    
    x=x_sorted;
    N_sort=round(N*rand*(percent));  % Number of population those will be eliminated from the group
    N_sort_Index=N-N_sort+1;     % Staring index number for elimination
    
    if length(LB)==1
        for kk=1:1:D
            lb_Ele(1:N_sort,kk)=LB;
            ub_Ele(1:N_sort,kk)=UB;
        end
    end
    
    
    if length(LB)~=1
        for kk=1:1:D
            lb_Ele(1:N_sort,kk)=LB(kk);
            ub_Ele(1:N_sort,kk)=UB(kk);
        end
    end
    
    x_born=lb_Ele+(ub_Ele-lb_Ele).*rand(N_sort,D);
    
    niiv=1;
    
    for iiv=N_sort_Index:1:N
        x(iiv,:)=x_born(niiv,:);
        iiv=iiv;
        niiv=niiv; 
        niiv=niiv+1;
    end
    
    clear lb_Ele;
    clear ub_Ele;
    clear x_born;
    clear iiv;
    
    for kk=1:1:D
        x(:,kk)=min(x(:,kk),ub(:,kk));
        x(:,kk)=max(x(:,kk),lb(:,kk));
    end
    
    %% Elimination Mechanism
    
    Fgbest_vect(it)=Alpha_score;
    F_g_bestDE_vect(it)=F_g_bestDE;
end

Fg_best=Alpha_score
gbest=Alpha_pos;


plot (Fgbest_vect);
hold on







break_point=1;











