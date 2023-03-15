clear all, close all, clc

%% ASSIGNMENT 1 - MVAR model

% data loading

load('Data_9channels.mat')

% number of channels

n_chan = size(Y,1);

% sampling time

Ts = 1/Fs;

% time vector

t = (0:Ts:Ts*length(Y)-Ts); % [s]

% flip the signal to have the last samples in the first column
Y = flip(Y')';

%% Two-fold Cross validation

% division of the dataset in two folds

cut = size(Y,2)/2;
Y_1 = Y(:,1:cut);       % fold_1
Y_2 = Y(:,cut+1:end);   % fold_2

% max model order

p_max = 10;

for p = 1:p_max
    % =====================================================================
    % Training on Y_1 
    Y_temp_1 = Y_1(:,1:end-p);  % support variable 
    
    n_sample = size(Y_temp_1,2);
    
    % building of matrix Phi_1
    Phi_1 = zeros(n_chan*p,n_sample);
    
    pos = 1;
    for j = 1:n_chan
        
        ind = 1;
        while ind < p+1
        
            Phi_1(pos+ind-1,:) = Y_1(j,ind+1:end-p+ind);
            ind = ind + 1;
        end % while
        
        pos = pos + ind -1;
    
    end % for
    
    % estimation of matrix A_1
    A_1 = Y_temp_1 * Phi_1' * inv(Phi_1*Phi_1');
    
    % =====================================================================
    % Training on Y_2 
    
    Y_temp_2 = Y_2(:,1:end-p);  % support variable
    
    n_sample = size(Y_temp_2,2);
    
    % building of matrix Phi_2
    
    Phi_2 = zeros(n_chan*p,n_sample);
    
    pos = 1;
    for j = 1:n_chan
        
        ind = 1;
        while ind < p+1
        
            Phi_2(pos+ind-1,:) = Y_2(j,ind+1:end-p+ind);
            ind = ind + 1;
        end % while
        
        pos = pos + ind -1;
    
    end % for
    
    % estimation of matrix A_2
    
    A_2 = Y_temp_2 * Phi_2' * inv(Phi_2*Phi_2');
    
    % =====================================================================
    % prediction of the model on test_2
    
    Y_hat_2 = A_1*Phi_2;
    
    % error of the model
    
    E_2 = Y_temp_2 - Y_hat_2;
    
    % determinant of the covariance matrix of the error
    
    Det_E_2 = det(cov(E_2'));
    
    % =====================================================================
    % prediction of the model on test_1
    
    Y_hat_1 = A_2*Phi_1;
    
    % error of the model
    
    E_1 = Y_temp_1 - Y_hat_1;
    
    % determinant of the covariance matrix of the error
    
    Det_E_1 = det(cov(E_1'));
    
    % =====================================================================
    % mean of the 2 fold
    
    Det_cv(p) = (Det_E_1 + Det_E_2) / 2;
   
end % for

% visualization 

figure()
hold on 
plot(1:1:p_max,Det_cv,'o-r')
ylabel('J(p)');
xlabel('Model order p');
title('Model order selection based on two-fold CV');

disp(['Model selection via two-fold CV']);
disp(['minimum value of J(p) = ', num2str(find(Det_cv==min(Det_cv)))]);
disp(['---------------------------------------------------------------']);

%% AIC and BIC

% max model order

p_max = 10;

for p = 1:p_max
    
    Y_temp = Y(:,1:end-p); % support variable
    
    n_sample = size(Y_temp,2);
    
    % building of matrix Phi
    Phi = zeros(n_chan*p,n_sample);
    
    pos = 1;
    for j = 1:n_chan
        
        ind = 1;
        while ind < p+1
        
            Phi(pos+ind-1,:) = Y(j,ind+1:end-p+ind);
            ind = ind + 1;
        end % while
        
        pos = pos + ind -1;
    
    end % for
    
    % estimation of matrix A
    
    A = Y_temp * Phi' * inv(Phi*Phi');
    
    % prediction of the model
    
    Y_hat = A*Phi;
    
    % error of the model
    
    E = Y_temp - Y_hat;
    
    % determinant of the covariance matrix of the error
    
    Det_E = det(cov(E'));
    
    % =====================================================================
    % Model order selection 
    
    % number of parameters
    
    D = n_chan^2 * p;
    
    % AIC
    
    AIC(p) = n_sample*log(Det_E) + 2*D;
    
    % BIC
    
    BIC(p) = n_sample*log(Det_E) + D*log(n_sample);
    
end % for

% visualization

figure()
hold on 
plot(1:1:p_max,AIC,'o-r')
plot(1:1:p_max,BIC,'o-b')
legend('AIC','BIC','Location','northwest');
xlabel('Model order p');
title('Model order selection based on statistical criteria');

disp(['Model selection via statistical criteria']);
disp(['minimum value of AIC(p)  = ', num2str(find(AIC==min(AIC)))]);
disp(['minimum value of BIC(p)  = ', num2str(find(BIC==min(BIC)))]);
disp(['---------------------------------------------------------------']);

%% fitting the selected model on whole dataset

% best model order
p_best = 3;

Y_temp = Y(:,1:end-p_best); % support variable
    
n_sample = size(Y_temp,2);
    
% building of matrix Phi
Phi = zeros(n_chan*p_best,n_sample);

pos = 1;
for j = 1:n_chan
        
    ind = 1;
    while ind < p_best+1
            Phi(pos+ind-1,:) = Y(j,ind+1:end-p_best+ind);
            ind = ind + 1;
    end % while
        
    pos = pos + ind -1;
    
end % for
    
% estimation of matrix A_best

A_best = Y_temp * Phi' * inv(Phi*Phi');

% prediction of the model
    
Y_hat = A_best*Phi;

% error of the model

E = Y_temp - Y_hat;

% covariance matrix of the error

Cov_E = cov(E');

%% ========================================================================
%%  ASSIGNMENT 2 - Connectivity measures

% Rearrange A_best as requiered by the function "fdMVAR"

A_best_new = zeros(size(A_best));

pos = 1;  % support variable

for i = 1:p_best
    
    % estraction of the columns of A_best related to the i-order
    col = (i:3:length(A_best));
    
    % filling of the new matrix A_best_new
    A_best_new(:,[pos:pos+length(col)-1]) = A_best(:,col);
    
    pos = pos + length(col);
end

% =========================================================================
% estimation of the several connectivity measurements

[dc,dtf,pdc,gpdc,coh,pcoh,pcoh2,h,s,pp,f] = fdMVAR(A_best_new,Cov_E,512,Fs);

COH = abs(coh) .^2;     % Coherence
PCOH = abs(pcoh) .^2;   % Partial Coherence 
DC = abs(dc) .^2;       % Directed Coherence
DTF = abs(dtf) .^2;     % Directed Transfer Function
GPDC = abs(gpdc) .^2;   % Generalized Partial Directed Coherence
PDC = abs(pdc) .^2;     % Partial Directed Coherence

% =========================================================================
% extraction of avereage connectivity matrices

% delta band D:[0.5 3] Hz
pos_0_5 = find(f == 0.5);   pos_3 = find(f == 3);
% theta band T:[4 8] Hz
pos_4 = find(f == 4);       pos_8 = find(f == 8);
% alpha band A:[8.5 12] Hz
pos_8_5 = find(f == 8.5);   pos_12 = find(f == 12);
% beta band B:[12.5 30] Hz
pos_12_5 = find(f == 12.5); pos_30 = find(f == 30);
% gamma band G:[30.5 60] Hz
pos_30_5 = find(f == 30.5); pos_60 = find(f == 60);

freq_pos = [pos_0_5,pos_3; pos_4,pos_8; pos_8_5,pos_12; pos_12_5,pos_30; pos_30_5,pos_60];

metrics = ["COH","PCOH","DC","DTF","GPDC","PDC"];
bands = ["delta","theta","alpha","beta","gamma"];

% structure that will contain all the average connectivity measurement for each frequency band
Avg = struct();

for j = 1:size(bands,2)
    for i=1:size(metrics,2)
        tmp = metrics(i);
        Avg.(metrics(i)).(bands(j)) = mean( eval(tmp+"(:,:,freq_pos(j,1):freq_pos(j,2))") ,3 );
        Avg.(metrics(i)).(bands(j))(logical(eye(n_chan))) = NaN;
    end
end

% =========================================================================
% visualization

for i = 1:size(metrics,2)
    max_val = max([max(Avg.(metrics(i)).(bands(1))), max(Avg.(metrics(i)).(bands(2))), max(Avg.(metrics(i)).(bands(3))), max(Avg.(metrics(i)).(bands(4))), max(Avg.(metrics(i)).(bands(5)))]);
    min_val = min([min(Avg.(metrics(i)).(bands(1))), min(Avg.(metrics(i)).(bands(2))), min(Avg.(metrics(i)).(bands(3))), min(Avg.(metrics(i)).(bands(4))), min(Avg.(metrics(i)).(bands(5)))]);
    
    fig = figure;
    tiledlayout(1,5)
    xticks = (1:1:n_chan);
    for j = 1:size(bands,2)
        h_fig(j) = nexttile;
        imagesc(Avg.(metrics(i)).(bands(j)),[min_val,max_val]);
        set(gca, 'XTick', xticks, 'XTickLabel', EEG_chan_name);
        set(gca, 'YTick', xticks, 'YTickLabel', EEG_chan_name);
        title(metrics(i)+' - '+ bands(j)+ ' band');
    end
    set(h_fig,'Colormap',turbo,'Clim',[min_val,max_val]);
    hcolorb = colorbar(h_fig(end));
    hcolorb.Layout.Tile = 'East';
end

% =========================================================================
% estimation of the total information outflow and inflow for each channel

% outflow --> DC and DTF

% DC
DC_outflow_F  = [sum(Avg.DC.delta(:,1),'omitnan'),sum(Avg.DC.theta(:,1),'omitnan'),sum(Avg.DC.alpha(:,1),'omitnan'),sum(Avg.DC.beta(:,1),'omitnan'),sum(Avg.DC.gamma(:,1),'omitnan')];
DC_outflow_FL = [sum(Avg.DC.delta(:,2),'omitnan'),sum(Avg.DC.theta(:,2),'omitnan'),sum(Avg.DC.alpha(:,2),'omitnan'),sum(Avg.DC.beta(:,2),'omitnan'),sum(Avg.DC.gamma(:,2),'omitnan')];
DC_outflow_FR = [sum(Avg.DC.delta(:,3),'omitnan'),sum(Avg.DC.theta(:,3),'omitnan'),sum(Avg.DC.alpha(:,3),'omitnan'),sum(Avg.DC.beta(:,3),'omitnan'),sum(Avg.DC.gamma(:,3),'omitnan')];
DC_outflow_C  = [sum(Avg.DC.delta(:,4),'omitnan'),sum(Avg.DC.theta(:,4),'omitnan'),sum(Avg.DC.alpha(:,4),'omitnan'),sum(Avg.DC.beta(:,4),'omitnan'),sum(Avg.DC.gamma(:,4),'omitnan')];
DC_outflow_CL = [sum(Avg.DC.delta(:,5),'omitnan'),sum(Avg.DC.theta(:,5),'omitnan'),sum(Avg.DC.alpha(:,5),'omitnan'),sum(Avg.DC.beta(:,5),'omitnan'),sum(Avg.DC.gamma(:,5),'omitnan')];
DC_outflow_CR = [sum(Avg.DC.delta(:,6),'omitnan'),sum(Avg.DC.theta(:,6),'omitnan'),sum(Avg.DC.alpha(:,6),'omitnan'),sum(Avg.DC.beta(:,6),'omitnan'),sum(Avg.DC.gamma(:,6),'omitnan')];
DC_outflow_P  = [sum(Avg.DC.delta(:,7),'omitnan'),sum(Avg.DC.theta(:,7),'omitnan'),sum(Avg.DC.alpha(:,7),'omitnan'),sum(Avg.DC.beta(:,7),'omitnan'),sum(Avg.DC.gamma(:,7),'omitnan')];
DC_outflow_PL = [sum(Avg.DC.delta(:,8),'omitnan'),sum(Avg.DC.theta(:,8),'omitnan'),sum(Avg.DC.alpha(:,8),'omitnan'),sum(Avg.DC.beta(:,8),'omitnan'),sum(Avg.DC.gamma(:,8),'omitnan')];
DC_outflow_PR = [sum(Avg.DC.delta(:,9),'omitnan'),sum(Avg.DC.theta(:,9),'omitnan'),sum(Avg.DC.alpha(:,9),'omitnan'),sum(Avg.DC.beta(:,9),'omitnan'),sum(Avg.DC.gamma(:,9),'omitnan')];
DC_outflow = [DC_outflow_F;DC_outflow_FL;DC_outflow_FR;DC_outflow_C;DC_outflow_CL;DC_outflow_CR;DC_outflow_P;DC_outflow_PL;DC_outflow_PR];

figure()
bar(DC_outflow)
legend(bands)
set(gca, 'XTick', xticks, 'XTickLabel', EEG_chan_name);
title('DC-based Outflow')

% DTF
DTF_outflow_F  = [sum(Avg.DTF.delta(:,1),'omitnan'),sum(Avg.DTF.theta(:,1),'omitnan'),sum(Avg.DTF.alpha(:,1),'omitnan'),sum(Avg.DTF.beta(:,1),'omitnan'),sum(Avg.DTF.gamma(:,1),'omitnan')];
DTF_outflow_FL = [sum(Avg.DTF.delta(:,2),'omitnan'),sum(Avg.DTF.theta(:,2),'omitnan'),sum(Avg.DTF.alpha(:,2),'omitnan'),sum(Avg.DTF.beta(:,2),'omitnan'),sum(Avg.DTF.gamma(:,2),'omitnan')];
DTF_outflow_FR = [sum(Avg.DTF.delta(:,3),'omitnan'),sum(Avg.DTF.theta(:,3),'omitnan'),sum(Avg.DTF.alpha(:,3),'omitnan'),sum(Avg.DTF.beta(:,3),'omitnan'),sum(Avg.DTF.gamma(:,3),'omitnan')];
DTF_outflow_C  = [sum(Avg.DTF.delta(:,4),'omitnan'),sum(Avg.DTF.theta(:,4),'omitnan'),sum(Avg.DTF.alpha(:,4),'omitnan'),sum(Avg.DTF.beta(:,4),'omitnan'),sum(Avg.DTF.gamma(:,4),'omitnan')];
DTF_outflow_CL = [sum(Avg.DTF.delta(:,5),'omitnan'),sum(Avg.DTF.theta(:,5),'omitnan'),sum(Avg.DTF.alpha(:,5),'omitnan'),sum(Avg.DTF.beta(:,5),'omitnan'),sum(Avg.DTF.gamma(:,5),'omitnan')];
DTF_outflow_CR = [sum(Avg.DTF.delta(:,6),'omitnan'),sum(Avg.DTF.theta(:,6),'omitnan'),sum(Avg.DTF.alpha(:,6),'omitnan'),sum(Avg.DTF.beta(:,6),'omitnan'),sum(Avg.DTF.gamma(:,6),'omitnan')];
DTF_outflow_P  = [sum(Avg.DTF.delta(:,7),'omitnan'),sum(Avg.DTF.theta(:,7),'omitnan'),sum(Avg.DTF.alpha(:,7),'omitnan'),sum(Avg.DTF.beta(:,7),'omitnan'),sum(Avg.DTF.gamma(:,7),'omitnan')];
DTF_outflow_PL = [sum(Avg.DTF.delta(:,8),'omitnan'),sum(Avg.DTF.theta(:,8),'omitnan'),sum(Avg.DTF.alpha(:,8),'omitnan'),sum(Avg.DTF.beta(:,8),'omitnan'),sum(Avg.DTF.gamma(:,8),'omitnan')];
DTF_outflow_PR = [sum(Avg.DTF.delta(:,9),'omitnan'),sum(Avg.DTF.theta(:,9),'omitnan'),sum(Avg.DTF.alpha(:,9),'omitnan'),sum(Avg.DTF.beta(:,9),'omitnan'),sum(Avg.DTF.gamma(:,9),'omitnan')];
DTF_outflow = [DTF_outflow_F;DTF_outflow_FL;DTF_outflow_FR;DTF_outflow_C;DTF_outflow_CL;DTF_outflow_CR;DTF_outflow_P;DTF_outflow_PL;DTF_outflow_PR];

figure()
bar(DTF_outflow)
legend(bands)
set(gca, 'XTick', xticks, 'XTickLabel', EEG_chan_name);
title('DTF-based Outflow')

% inflow --> GPDC and PDC

% GPDC
GPDC_inflow_F  = [sum(Avg.GPDC.delta(1,:),'omitnan'),sum(Avg.GPDC.theta(1,:),'omitnan'),sum(Avg.GPDC.alpha(1,:),'omitnan'),sum(Avg.GPDC.beta(1,:),'omitnan'),sum(Avg.GPDC.gamma(1,:),'omitnan')];
GPDC_inflow_FL = [sum(Avg.GPDC.delta(2,:),'omitnan'),sum(Avg.GPDC.theta(2,:),'omitnan'),sum(Avg.GPDC.alpha(2,:),'omitnan'),sum(Avg.GPDC.beta(2,:),'omitnan'),sum(Avg.GPDC.gamma(2,:),'omitnan')];
GPDC_inflow_FR = [sum(Avg.GPDC.delta(3,:),'omitnan'),sum(Avg.GPDC.theta(3,:),'omitnan'),sum(Avg.GPDC.alpha(3,:),'omitnan'),sum(Avg.GPDC.beta(3,:),'omitnan'),sum(Avg.GPDC.gamma(3,:),'omitnan')];
GPDC_inflow_C  = [sum(Avg.GPDC.delta(4,:),'omitnan'),sum(Avg.GPDC.theta(4,:),'omitnan'),sum(Avg.GPDC.alpha(4,:),'omitnan'),sum(Avg.GPDC.beta(4,:),'omitnan'),sum(Avg.GPDC.gamma(4,:),'omitnan')];
GPDC_inflow_CL = [sum(Avg.GPDC.delta(5,:),'omitnan'),sum(Avg.GPDC.theta(5,:),'omitnan'),sum(Avg.GPDC.alpha(5,:),'omitnan'),sum(Avg.GPDC.beta(5,:),'omitnan'),sum(Avg.GPDC.gamma(5,:),'omitnan')];
GPDC_inflow_CR = [sum(Avg.GPDC.delta(6,:),'omitnan'),sum(Avg.GPDC.theta(6,:),'omitnan'),sum(Avg.GPDC.alpha(6,:),'omitnan'),sum(Avg.GPDC.beta(6,:),'omitnan'),sum(Avg.GPDC.gamma(6,:),'omitnan')];
GPDC_inflow_P  = [sum(Avg.GPDC.delta(7,:),'omitnan'),sum(Avg.GPDC.theta(7,:),'omitnan'),sum(Avg.GPDC.alpha(7,:),'omitnan'),sum(Avg.GPDC.beta(7,:),'omitnan'),sum(Avg.GPDC.gamma(7,:),'omitnan')];
GPDC_inflow_PL = [sum(Avg.GPDC.delta(8,:),'omitnan'),sum(Avg.GPDC.theta(8,:),'omitnan'),sum(Avg.GPDC.alpha(8,:),'omitnan'),sum(Avg.GPDC.beta(8,:),'omitnan'),sum(Avg.GPDC.gamma(8,:),'omitnan')];
GPDC_inflow_PR = [sum(Avg.GPDC.delta(9,:),'omitnan'),sum(Avg.GPDC.theta(9,:),'omitnan'),sum(Avg.GPDC.alpha(9,:),'omitnan'),sum(Avg.GPDC.beta(9,:),'omitnan'),sum(Avg.GPDC.gamma(9,:),'omitnan')];
GPDC_inflow = [GPDC_inflow_F;GPDC_inflow_FL;GPDC_inflow_FR;GPDC_inflow_C;GPDC_inflow_CL;GPDC_inflow_CR;GPDC_inflow_P;GPDC_inflow_PL;GPDC_inflow_PR];

figure()
bar(GPDC_inflow)
legend(bands)
set(gca, 'XTick', xticks, 'XTickLabel', EEG_chan_name);
title('GPDC-based Inflow')

% PDC
PDC_inflow_F  = [sum(Avg.PDC.delta(1,:),'omitnan'),sum(Avg.PDC.theta(1,:),'omitnan'),sum(Avg.PDC.alpha(1,:),'omitnan'),sum(Avg.PDC.beta(1,:),'omitnan'),sum(Avg.PDC.gamma(1,:),'omitnan')];
PDC_inflow_FL = [sum(Avg.PDC.delta(2,:),'omitnan'),sum(Avg.PDC.theta(2,:),'omitnan'),sum(Avg.PDC.alpha(2,:),'omitnan'),sum(Avg.PDC.beta(2,:),'omitnan'),sum(Avg.PDC.gamma(2,:),'omitnan')];
PDC_inflow_FR = [sum(Avg.PDC.delta(3,:),'omitnan'),sum(Avg.PDC.theta(3,:),'omitnan'),sum(Avg.PDC.alpha(3,:),'omitnan'),sum(Avg.PDC.beta(3,:),'omitnan'),sum(Avg.PDC.gamma(3,:),'omitnan')];
PDC_inflow_C  = [sum(Avg.PDC.delta(4,:),'omitnan'),sum(Avg.PDC.theta(4,:),'omitnan'),sum(Avg.PDC.alpha(4,:),'omitnan'),sum(Avg.PDC.beta(4,:),'omitnan'),sum(Avg.PDC.gamma(4,:),'omitnan')];
PDC_inflow_CL = [sum(Avg.PDC.delta(5,:),'omitnan'),sum(Avg.PDC.theta(5,:),'omitnan'),sum(Avg.PDC.alpha(5,:),'omitnan'),sum(Avg.PDC.beta(5,:),'omitnan'),sum(Avg.PDC.gamma(5,:),'omitnan')];
PDC_inflow_CR = [sum(Avg.PDC.delta(6,:),'omitnan'),sum(Avg.PDC.theta(6,:),'omitnan'),sum(Avg.PDC.alpha(6,:),'omitnan'),sum(Avg.PDC.beta(6,:),'omitnan'),sum(Avg.PDC.gamma(6,:),'omitnan')];
PDC_inflow_P  = [sum(Avg.PDC.delta(7,:),'omitnan'),sum(Avg.PDC.theta(7,:),'omitnan'),sum(Avg.PDC.alpha(7,:),'omitnan'),sum(Avg.PDC.beta(7,:),'omitnan'),sum(Avg.PDC.gamma(7,:),'omitnan')];
PDC_inflow_PL = [sum(Avg.PDC.delta(8,:),'omitnan'),sum(Avg.PDC.theta(8,:),'omitnan'),sum(Avg.PDC.alpha(8,:),'omitnan'),sum(Avg.PDC.beta(8,:),'omitnan'),sum(Avg.PDC.gamma(8,:),'omitnan')];
PDC_inflow_PR = [sum(Avg.PDC.delta(9,:),'omitnan'),sum(Avg.PDC.theta(9,:),'omitnan'),sum(Avg.PDC.alpha(9,:),'omitnan'),sum(Avg.PDC.beta(9,:),'omitnan'),sum(Avg.PDC.gamma(9,:),'omitnan')];
PDC_inflow = [PDC_inflow_F;PDC_inflow_FL;PDC_inflow_FR;PDC_inflow_C;PDC_inflow_CL;PDC_inflow_CR;PDC_inflow_P;PDC_inflow_PL;PDC_inflow_PR];

figure()
bar(PDC_inflow)
legend(bands)
set(gca, 'XTick', xticks, 'XTickLabel', EEG_chan_name);
title('PDC-based Inflow')