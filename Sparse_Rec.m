% Floris van Leeuwen student nr:4676092
% Liam Rees          student nr:5735726

clear
close all

% Set hyperparameters
Num_meas = 10; %Number of measurements
It_Ista = 10000; %Iterations for ISTA
It_SDP = 50000; %Iterations for SDP

% import data
h = importdata('EE4740_Proj7_Dataset_CPR.mat');

% Set basis Parameters
N = 32;
% Commonly used values
U_N = dctmtx(N);
x = U_N * h;


%% Question 1

% Get y and A
[y_m, A] = GetyA(Num_meas, x);

% Recovery without loss of sign info
z = ISTA(y_m, A, It_Ista);
h_rec = U_N' * z;
MSE_rec = mean((h_rec - h).^2);

% Recovery with loss of sign info
y_abs = abs(y_m);
z_abs = ISTA(y_abs, A, It_Ista);
h_abs = U_N' * z_abs;
MSE_abs = mean((h_abs - h).^2);


%figures question 1
figure
hold on
semilogy(MSE_rec,'LineWidth',1.2,'Color',[0 0.4470 0.7410]); %MSE
semilogy(MSE_abs,'LineWidth',1.2,'Color',[0.8500 0.3250 0.0980])

semilogy(mean(MSE_rec)*ones(1,100),':','Color',[0 0.4470 0.7410],'LineWidth',1.2) %Mean MSE
semilogy(mean(MSE_abs)*ones(1,100),':','Color',[0.8500 0.3250 0.0980],'LineWidth',1.2)

legend('L1 Ideal','L1 No Sign Info')
title('Mean Squared Error, for reconstructed L1 methods')
ylabel('MSE')
xlabel('signal')
grid on
hold off


%% Question 2

% Recovery without need for  sign info
x = SDB(y_abs, A, It_SDP);
h_sdp = U_N' * x;
MSE_SDP = mean((abs(h)-abs(h_sdp)).^2);

%figures

figure
plot(10*log10(MSE_rec)); hold on
plot(10*log10(MSE_abs));
plot(10*log10(MSE_SDP));
legend('L1 Ideal','L1 No Sign Info','Projected Subgradient. No Sign Info')
hold off

figure
hold on
semilogy(MSE_rec,'LineWidth',1.2,'Color',[0 0.4470 0.7410]); %MSE
semilogy(MSE_abs,'LineWidth',1.2,'Color',[0.8500 0.3250 0.0980]);
semilogy(MSE_SDP,'LineWidth',1.2,'Color',[0.4940 0.1840 0.5560]);

semilogy(mean(MSE_rec)*ones(1,100),':','Color',[0 0.4470 0.7410],'LineWidth',1.2); %Mean MSE
semilogy(mean(MSE_abs)*ones(1,100),':','Color',[0.8500 0.3250 0.0980],'LineWidth',1.2);
semilogy(mean(MSE_SDP)*ones(1,100),':','Color',[0.4940 0.1840 0.5560],'LineWidth',1.2);

legend('L1 Ideal','L1 No Sign Info','Projected Subgradient. No Sign Info')
ylabel('MSE')
xlabel('signal')
grid on
hold off


%% functions
function [y_m, A] = GetyA(Meas_size, X)

%Define Size
[N, m] = size(X);

%Define DCT
DCT = dctmtx(N);

% Initialize arrays to store y and A
y_m = zeros(Meas_size, m);
A = zeros(Meas_size, N);

for k = 1:Meas_size

    Pm = randn(N,1); %setting up Pm
    A(k,:) = Pm'*DCT; %A stays the same for every h

    for i = 1:m

        y_m(k,i) = A(k,:)*X(:,i); %measurement

    end

end

end



function [z] = ISTA(y, A, iterations )

%Define sizes
 m = size(y, 2);
 N = size(A,2);

%Define nu and lambda
nu = 1/6000; % < min size
lambda = 1; % Gave better results than lambda = 1

% Initialize arrays to store z
z = zeros(N,m);

for i = 1:m

    for k = 1:iterations

        z(:, i) = prox(z(:, i)-2*nu*A'*(A*z(:, i)-y(:,i)),lambda,nu); %Gradient step with prox operation

    end
end 

end



function u_opt = prox(x,lamda,nu)

u_opt = x;

%Prox operation 
for i = 1:length(x)
    if(x(i) > (lamda*nu))
        u_opt(i) =  x(i) - lamda*nu;

    elseif (x(i) < (-1*lamda*nu))
        u_opt(i) =  x(i) + lamda*nu;

    else
        u_opt(i) = 0;
    end

end

end



function [x] = SDB(y_m, A, It_SDP)

% Parameters
lambda = 1;

% Gradient function
g = @(X)  eye(size(X,1)) +lambda*sign(X); 

% Relevant Sizes
size_h = size(y_m, 2);
[Num_meas, N] = size(A);

% Initialize X
X = 0.1*randn(size(A,2));
x = zeros(N, size_h);

% bi for constraint 
bi = y_m.*conj(y_m);

% A for constraint bi = A*x
Phi = zeros(N, N, Num_meas);
Atr = ctranspose(A);
for i = 1:1:Num_meas
    Phi(:,:,i) = Atr(:,i)*ctranspose(Atr(:,i));
end
Phi_vec = reshape(Phi,[1024 Num_meas]);
A_new = Phi_vec';

% SDP
for i = 1:size_h

disp("Current column of h:" + i)

    for k = 1:It_SDP
        
        Beta = 12000;
        % Compute subgradient
        G =  g(X);
        
        % Varying Beta over iterations
        if k > 0.2*It_SDP
           Beta = 12000 + ((k-0.2*It_SDP)/(0.8*It_SDP))*26000;
        end 

        % Update X
        X = X - 1/Beta*G;
    
        % Vectorize X
        VecX = reshape(X,1,[]);

        % Projection step constraint 1: bi = A*x
        X_new = VecX' - A_new'*pinv(A_new*A_new')*(A_new*VecX'-bi(:,i));

        % Reshape X
        X= reshape(X_new,[32 32]);
    
        % Projection step constraint 2: x=> 0
        [V, D] = eig((X + X')/2);
        D = max(D,0);
        X = V * D * V';
        
    end

% Retrieve vector form X
[u, s, ~]=svds(X,1);

% Get h
x(:,i)=u*sqrt(s);

end

end
