%% Developed by: Taniel Winner
% This script generates the figures associated with manuscript: XXX
% Date: 03/21/24

%% Set figure parameters
set(groot,'DefaultFigureColor','w')
set(groot,'DefaultAxesFontSize',18)
% set(groot,'DefaultAxesLineWidth',.5)
set(groot,'DefaultAxesFontName','Arial')
% set(groot,'DefaultLineLineWidth',1)
set(groot,'DefaultFigureUnits','inches')
set(groot,'DefaultFigurePosition',[.5, .5, 3.5, 2.5])

%% Load in labels for trials

load('subjects_label.mat');
load('sub_num_label.mat');
load('speeds_label.mat');
load('trialtype.mat');

% Load raw extracted data used to train RNN models
%load('ExtractedData.mat')
load('DataCell_3D_kinematics_clean.mat')
load('DataCell_kinetics_clean.mat')


% Load phase-averaged raw data and gait phase labels
load('Alldata_phaseavg_nopca_031624_PhaseAvgPcs_shift_LHS.mat')
load('Alldata_phaseavg_nopca_031624_ShiftGaitPhase_LHS.mat')

% Load in discrete variables table
load('HYA_Study_InformationTable_120523.mat')

% Load in 26 commonly used discrete variables
DiscreteVars = load('YA_26Discrete_numarray.mat');
load('YA_26Discrete_Table.mat');



%% Load the generated Gait Signatures (from Python) and associated info here 

GS_L = 1024; %total dynamics features

%2D Kinematics
GS_2Dkin_gaitphase = load('2Dkinematics_gaitsigs_ShiftGaitPhase_LHS.mat');
GS_2Dkin_pcvar = load('2Dkinematics_gaitsigs_pca_var.mat');
GS_2Dkin_sigs = load('2Dkinematics_gaitsigs_Gaitsignatures_shift.mat');


% %3D Kinematics
GS_3Dkin_gaitphase = load('3Dkinematics_gaitsigs_ShiftGaitPhase_LHS.mat');
GS_3Dkin_sigs = load('3Dkinematics_gaitsigs_Gaitsignatures_shift.mat');
GS_3Dkin_pcvar = load('3Dkinematics_gaitsigs_pca_var.mat');


% Kinetics
GS_kinetics_gaitphase = load('kinetics_UPDATE_gaitsigs_reg_adam_LB499_031524_ShiftGaitPhase_LHS.mat');
GS_kinetics_pcvar = load('kinetics_UPDATE_gaitsigs_reg_adam_LB499_031524_pca_var.mat');
GS_kinetics_sigs = load('kinetics_UPDATE_gaitsigs_reg_adam_LB499_031524_Gaitsignatures_shift.mat');


% All Data
GS_alldata_gaitphase = load('alldata_gaitsigs_reg_adam_LB499_031624_ShiftGaitPhase_LHS.mat');
GS_alldata_sigs = load('alldata_gaitsigs_reg_adam_LB499_031624_Gaitsignatures_shift.mat');
GS_alldata_pcvar = load('alldata_gaitsigs_reg_adam_LB499_031624_pca_var.mat');

%% Store all Gait Signature Data in a Cell Array for Access by Functions Later
% Order as 4 x 1 cell array where the cols represent: 1) 2D kinematics, 2)
% 3D kinematics, 3) kinetics, 4) all dat for RNN gait sig

% These specified index values of PCs explain at least 80% of the variance explained in the original 
% data features and maximizes the classification accuracy across individuals
% * each PC is 100 samples long and concatenated one after the next in
% order of dominance (most variance explained)

Sigs = cell(1,4);
Sigs{1,1} = GS_2Dkin_sigs.GaitSigs(:,1:600); % 6 PCs
Sigs{1,2} = GS_3Dkin_sigs.GaitSigs(:,1:1000); % 10 PCs
Sigs{1,3} = GS_kinetics_sigs.GaitSigs(:,1:2900); % 29 PCs
Sigs{1,4} = GS_alldata_sigs.GaitSigs(:,1:3000); % 30 PCs

% Store all signatures in a single table
Signatures = cell2table(Sigs, 'VariableNames', {'GS_2Dkin', 'GS_3Dkin', 'GS_kinetics', 'GS_alldata'});

% Access each signature as follows:
% Signatures.GS_2Dkin{1,1}

% Total Variability of features accounted for by first 6 PCs (For gait signatures
% it is what is accounted for in the dynamics)
kinetics_cumvar = load('kinetics_UPDATE_gaitsigs_reg_adam_LB499_031524Cumulative_VAF_all.mat');
kin3D_cumvar = load('3Dkinematics_gaitsigs_reg_adam_LB499_110523Cumulative_VAF.mat');
kin2D_cumvar = load('2Dkinematics_gaitsigs_reg_adam_LB499_110423Cumulative_VAF.mat');
alldata_cumvar = load('alldata_gaitsigs_reg_adam_LB499_031624Cumulative_VAF_all.mat');

VAF = cell(1,4);
VAF{1,1} = sum(kin2D_cumvar.Cumulative_VAF(6));
VAF{1,2} = sum(kin3D_cumvar.Cumulative_VAF(6));
VAF{1,3} = sum(kinetics_cumvar.Cumulative_VAF(6));
VAF{1,4} = sum(alldata_cumvar.Cumulative_VAF(6));


VarianceAF_6PCs = cell2table(VAF, 'VariableNames', {'GS_2Dkin', 'GS_3Dkin', 'GS_kinetics', 'GS_alldata'});


% Cumulative VAR across all PCs (at each PC)
CUM_VAF = cell(1,4);
CUM_VAF{1,1} = kin2D_cumvar.Cumulative_VAF;
CUM_VAF{1,2} = kin3D_cumvar.Cumulative_VAF;
CUM_VAF{1,3} = kinetics_cumvar.Cumulative_VAF;
CUM_VAF{1,4} = alldata_cumvar.Cumulative_VAF;

CUM_VarianceAF_allPCs = cell2table(CUM_VAF, 'VariableNames', {'GS_2Dkin', 'GS_3Dkin', 'GS_kinetics', 'GS_alldata'});


% Table of Gait Phases 
GaitPhase = cell(1,4);
GaitPhase{1,1} = GS_2Dkin_gaitphase.GaitPhase_shift_LHS;
GaitPhase{1,2} = GS_3Dkin_gaitphase.GaitPhase_shift_LHS;
GaitPhase{1,3} = GS_kinetics_gaitphase.GaitPhase_shift_LHS;
GaitPhase{1,4} = GS_alldata_gaitphase.GaitPhase_shift_LHS;


% Store all signatures in a single table
GaitPhaseTable = cell2table(GaitPhase, 'VariableNames', {'GS_2Dkin', 'GS_3Dkin', 'GS_kinetics', 'GS_alldata'});


%% Visualization: Representative Plots of Raw Data 

% Plotting Representative 3D kinematics

%Need: Y10(orange --> green)- sub8, Y17 (olive --> orange)--sub14,  and Y16 (purple -->
%purple)--sub13

kinematic3Dlabels = {'RHip_x','RHip_y','RHip_z','RKnee_x', 'RKnee_y','RKnee_z','RAnk_x','RAnk_y','RAnk_z','LHip_x','LHip_y','LHip_z','LKnee_x','LKnee_y','LKnee_z','LAnk_x','LAnk_y','LAnk_z'};

trialsamples = 500;
indices = [1,4,7,8,9]; %features
subject = 8; %8,14,13
condition = 8; % 1- extreme slow, 3 - self-selected TM, 8 - walk to run transition
figure()
for i = 1:length(indices)
    feat = indices(i);
    subplot(length(indices),1,i)    
    plot(DataCell_3D_kinematics_clean{subject,condition}(feat,1:trialsamples), 'Linewidth',1)
    ylabel(kinematic3Dlabels{feat});
    %ylim([-30,60])
end

% Plotting representative kinetic data 

%Need: Y10(orange --> green)- sub8, Y17 (olive --> orange)--sub14,  and Y16 (purple -->
%purple)--sub13

kinetics3Dlabels =  {"R GRFX","R GRFY","R GRFZ","R Hip MomX","R Hip MomY","R Hip MomZ","R Knee MomX","R Knee MomY","R Knee MomZ","R Ank MomX","R Ank MomY","R Ank MomZ",...
    "R Hip PowX","R Hip PowY","R Hip PowZ","R Knee PowX","R Knee PowY","R Knee PowZ","R Ank PowX","R Ank PowY","R Ank PowZ",...
    "L GRFX","L GRFY","L GRFZ","L Hip MomX","L Hip MomY","L Hip MomZ","L Knee MomX","L Knee MomY","L Knee MomZ","L Ank MomX","L Ank MomY","L Ank MomZ",...
    "L Hip PowX","L Hip PowY","L Hip PowZ","L Knee PowX","L Knee PowY","L Knee PowZ","L Ank PowX","L Ank PowY","L Ank PowZ"}; 

indices = [3,4,7,10,19]; %features
subject = 14; %8,14,13
condition = 8; % 1- extreme slow, 3 - self-selected TM, 8 - walk to run transition
figure()
for i = 1:length(indices)
    feat = indices(i);
    subplot(length(indices),1,i)    
    plot(DataCell_kinetics_clean{subject,condition}(feat,1:trialsamples), 'Linewidth',1)
    ylabel(kinetics3Dlabels{feat});
    %ylim([-30,60])
end

%% Plot 3 speeds for a single feature - Right Angle Flexion/Extension
trialsamples = 1000;
feat = 7; %right ankle
indices = [1]; %features
subject = 8; %8,14,13
condition = [1,3,8]; % 1- extreme slow, 3 - self-selected TM, 8 - walk to run transition
figure()
for p = 1:3    
    subplot(3,1,p)
    plot(DataCell_3D_kinematics_clean{subject,condition(p)}(feat,1:trialsamples), 'Linewidth',2)
    ylim([-35 10])
end
    
% Plot 3 speeds for Vertical GRF

trialsamples = 1000;
feat = 3; %right ankle
indices = [1]; %features
subject = 8; %8,14,13
condition = [1,3,8]; % 1- extreme slow, 3 - self-selected TM, 8 - walk to run transition
figure()
for p = 1:3    
    subplot(3,1,p)
    plot(DataCell_kinetics_clean{subject,condition(p)}(feat,1:trialsamples), 'Linewidth',2)
    ylim([0 1.8])
end


%% Figure 1: Phase averaged data - shifted to start at Left Heel Strike

% LTO = 1;
% LHS = 2;
% RTO = 3;
% RHS = 4;

all_labels = horzcat(kinematic3Dlabels,kinetics3Dlabels);

green_gradient = [
247,252,245
229,245,224
199,233,192
161,217,155
116,196,118
65,171,93
35,139,69
0,109,44
0,68,27
]./255;

orange_gradient = [
255,255,204
255,237,160
254,217,118
254,178,76
253,141,60
252,78,42
227,26,28
189,0,38
128,0,38 % Darkest orange
]./255;

purple_gradient = [
247,252,253
224,236,244
191,211,230
158,188,218
140,150,198
140,107,177
136,65,157
129,15,124
77,0,75
]./255;

ColorCell = {};
ColorCell{1,1} = green_gradient;
ColorCell{1,2} = orange_gradient;
ColorCell{1,3} = purple_gradient;

subjects = ["YA010","YA017","YA016"];
colors = [0 1 0; 1 0.5 0; 0.5 0 0.5];
for feat = 1:60;
    figure()
    for i = 1:3
        label = subjects(i);
        indices = find(strcmp(SubjectTrainTrials, label));
        for j = 1:length(indices)

            gaitphaselabel = GaitPhase_shift_LHS(indices(j),:);
            f_lhs = min(find(gaitphaselabel == 2));
            gaitphaselabel_lhsstart = circshift(gaitphaselabel,-f_lhs);
            trace = squeeze(PhaseAvgPCs_shift_LHS(indices(j),feat,:));
            trace_lhs = circshift(trace, -f_lhs);
            plot(trace_lhs,'Color',ColorCell{1,i}(j,:),'LineWidth',2); %gradient of color across subject speed trials
                                 
            %plot(trace_lhs,'Color',colors(i,:),'LineWidth',2) % single
            %color per all subject trials -- no gradient

            hold on
        end
    end
    title(all_labels{feat})
end

%% Figure 2: Pipeline figure adapted from previous paper materials and manipulated plots (Gait signature loops and MDS plots) generated later 
% ~~~~ 
%% Figure 3: Plot Scree Plots for all Models to Determine the optimal # PCs to use for each. 
% Gait Sigs all data and kinetics fail to reach 75% with over 40PCs. 
labels = {'GS 2Dkin', 'GS 3Dkin', 'GS kinetics', 'GS All Data'};
for c = 1:4 % number of models   
    y = CUM_VAF{1,c};
    x = 1:length(y); %60 features for all data
    L = length(y);
    figure()
    plot(x(1:L),y(1:L)*100,'k','LineWidth',4)
    hold on
    plot(x(1:L),y(1:L)*100,'rd','LineWidth',2)

    xlabel('Number of Principal Components')
    ylabel('Variance Explained (%)')
    ylim([0,100])
    title(labels{c})
end

%% Figure 4: Pipeline figure of SVM classification classification
% ~~~~~~
%% Figure 5 and 6: Plot visualization of all signatures
% This function plots the 3D gait signature loops across model types
% colored by gait phase, individual and speed
ModelsToPlot = [1:4]; %list the model indices that you would like to be plotted
AllModels_GaitSignatureVisulizations(Signatures,ModelsToPlot,GaitPhaseTable)

%% Plot 3D MDS of all individuals signatures across data types

% [Action Required]: Load in the specific gait signature data type below:
GaitSigs = Signatures.GS_3Dkin{1,1};


custom_colors =  [
    0, 0, 1;     % Blue - change?! same as Royal Blue
    1, 0, 0;     % Red
    0, 1, 1;     % Cyan
    1, 0, 1;     % Magenta
    0.5, 0, 0;   % Maroon
    1, 1, 0;     % Yellow
    0, 0, 0;     % Black
     0, 0, 1;    % Royal Blue
    0, 0, 0.5;   % Navy
    1, 0.5, 0;   % Orange
     0.3,0,0.5;   % DarkIndigo  
    0, 0.5, 0;   % Dark Green
    0, 0.5, 1;   % Light Blue
    1, 0, 0.5;   % Dark Pink
    0.5, 0.5, 1; % Light Indigo
    0.8, 0.6, 0.9; % Light Purple
    0.5, 0.5, 0; % Olive Green
    1, 0.5, 0.5; % Light Pink
    0.9, 0.9, 0.6; % Khaki
    0.5, 1, 0.5; % Light Green
];

All_GS = [GaitSigs];
all_speed = [SpeedTrain];  % Append 0 speed for the centroid
speed = SpeedTrain;


% Create a dissimilarity matrix
D = pdist2(All_GS, All_GS, 'euclidean'); % Dissimilarity matrix
Y3 = mdscale(D, 3); % Get the 3D coordinates for plotting

% Define marker size range
minSize = 75;
maxSize =700;

% Calculate marker size based on speed
markerSize = minSize + (speed - min(speed)) .* ((maxSize - minSize) / (max(speed) - min(speed)));

% Map labels to corresponding colors
pointColors = custom_colors(sub_num, :);   % this maps a color to each SUBID uniquely for the length of all trials. 

% Scatter plot of points with size and color according to speed (ignore the
% centroid)
figure()
scatter3(Y3(1:end, 1), Y3(1:end, 2), Y3(1:end, 3), markerSize, pointColors, 'filled','MarkerEdgeColor', 'k','HandleVisibility', 'off');

%labels = unique(SubjectTrainTrials);

hold on;

% Manually add an overall legend for each color
%colors(2,:) = [];

legendlabels = {'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11', 'Y12', 'Y14', 'Y15', 'Y16', 'Y17', 'Y18', 'Y19', 'Y20'};

 f = find(trialtype == 1);
 colors_f = pointColors(f,:);

for i = 1:size(colors_f, 1)
    scatter3(nan, nan, nan, 50, colors_f(i, :), 'filled');
end
legend(legendlabels, 'Location', 'eastoutside');


% Set axis equal and enable grid
axis equal;
grid on;
grid(gca, 'minor');

% Label the axes
xlabel('MDS 1');
ylabel('MDS 2');
zlabel('MDS 3');

% Add a title
title('Gait Signatures of Healthy Young Adults Across Speeds');

%% Plot MDS by speed 

% [Action Required]: Load in the specific gait signature data type below:
GaitSigs = Signatures.GS_3Dkin{1,1};

All_GS = [GaitSigs];
conditions = SpeedTrain*0.01;
% Create a dissimilarity matrix.
D = pdist2(All_GS,All_GS,'euclidean'); % dissimilarity matrix
Y2 = mdscale(D,3); % get the centroid in 3D for plotting

rng default % for reproducibility
U = unique(conditions);
% get distinguishable colors
n_colors = length(U);
colors = jet(n_colors);

figure()
c = 0;
for n = 1:length(SpeedTrain) % number of individuals
    c = c+1;
    f = find(conditions(n) == U); % find the index color to use
    c1 = colors(f,1);
    c2 = colors(f,2);
    c3 = colors(f,3);
    S = c;
    scatter3(Y2(n,1),Y2(n,2),Y2(n,3),400,'MarkerFaceColor',[c1,c2,c3], 'MarkerEdgeColor',[0,0,0],'MarkerEdgeColor',[0,0,0]); % specify color and markersize
    hold on


end
axis equal
grid on
xlabel('MDS 1');
ylabel('MDS 2');
zlabel('MDS 3');
%grid(gca,'minor')
c = colorbar();
caxis([min(conditions) max(conditions)]);
c.Location = 'eastoutside';

%% Individual classification task using support vector machine classifiers
% This function generates accuracies across 140 model runs for each data
% type and varying the number of training speed trials in each test from 1 to 8
% speed trials

[AccuracyDistribution] = Plot_VariedHoldOutSet_IndivClassification(Signatures);


%% Test effect sizes 
%effect size r less than 0.3 → small effect
%effect size r between 0.3 and 0.5 → medium effect
%effect size r greater than 0.5 → large effect

P_store = [];
MannWhit_r = [];


for i = 1:8

    group1 = AccuracyDistribution{i,1}; %2D kinematics signatures
    group2 = AccuracyDistribution{i,2}; %3D kinematic signatures

    % Perform Mann-Whitney U test
    [p, ~, stats] = ranksum(group1, group2);
    
    % Get Z-score from Mann-Whitney U test
    Z = stats.zval;
    
    % Compute sample sizes
    N1 = numel(group1);
    N2 = numel(group2);
    
    % Compute effect size (r)
    r = Z / sqrt(N1);
    
    % Display results
    P_store = [P_store; p];
    MannWhit_r = [MannWhit_r; r];

    display('done')

end


%%  Confusion Matrix for 2D kinematics using 4 speed trials 

% Remove these subjects --> after removal subjects (n = 14)
YA004_indices = find(SubjectTrainTrials == 'YA004');
YA015_indices = find(SubjectTrainTrials == 'YA015');
YA006_indices = find(SubjectTrainTrials == 'YA006');

remove_indices = [YA004_indices;YA015_indices;YA006_indices];

% Create a logical index for elements to keep
keep_indices = true(size(SubjectTrainTrials));
keep_indices(remove_indices) = false;
num_runs = 10; 
num_trials_per_sub = 9;
num_subjects = 14;
num_training_trials = 4;
average_accuracy = [];
std_accuracy = [];


GaitSigs = Signatures.GS_2Dkin{1,1}; % specify signature type here
Altered_SubjLabels = SubjectTrainTrials(keep_indices);
Altered_GaitSigs = GaitSigs(keep_indices,:);
Altered_sub_num = sub_num(keep_indices);

accuracy_results = zeros(num_runs,1);

% Create a template for a linear SVM classifier
template = templateSVM('Standardize', 1); % Standardize features

for run = 1:num_runs
    train_data = [];
    train_labels = [];
    test_data = [];
    test_labels = [];


    %rng(run); %set a random seed for reproducibility

    subs = unique(Altered_sub_num);

    for subject = 1:num_subjects
        rng(run); % Set a random seed for reproducibility each run
        shuffled_ind = randperm(num_trials_per_sub); %randomizes which speed trials each subject and run

        train_indices = shuffled_ind(1:num_training_trials);
        test_indices = shuffled_ind(num_training_trials + 1:num_trials_per_sub);

        sub_ind = find(Altered_sub_num == subs(subject)==1);

        train_trial_indices = sub_ind(train_indices);
        test_trial_indices = sub_ind(test_indices);

        test_data = [test_data;Altered_GaitSigs(test_trial_indices, :)]; %concatenate each subject's test data
        test_labels = [test_labels;Altered_sub_num(test_trial_indices)];

        train_data = [train_data;Altered_GaitSigs(train_trial_indices, :)]; %concatenate each subject's test data
        train_labels = [train_labels;Altered_sub_num(train_trial_indices)];

        % Train an SVM classifier on the training data
        svm_model = fitcecoc(train_data, train_labels, 'Learners', template, 'Coding', 'onevsall');
        % Predict labels for the test data
        predicted_labels = predict(svm_model, test_data);

        % Calculate accuracy for this fold
        accuracy = sum(predicted_labels == test_labels) / numel(test_labels);

        % Store accuracy in results
        accuracy_results(subject, 1) = accuracy;
    end

end

average_accuracy = [average_accuracy mean(accuracy_results(:))];
std_accuracy = [std_accuracy std(accuracy_results(:))];


% Construct the confusion matrix
conf_matrix = confusionmat(test_labels, predicted_labels)
labels = unique(test_labels);

% Plot the confusion matrix
figure;
heatmap(labels, labels, conf_matrix, 'Colormap', jet, 'ColorbarVisible', 'on', 'FontSize', 8, 'XLabel', 'Predicted Labels', 'YLabel', 'True Labels', 'Title', 'Confusion Matrix');

%% Intra and Inter-individual 3D distance histogram generation

points = Y3; % Set the points to the MDS coordinates (generated above)

labels = SubjectTrainTrials;


% Initialize arrays for intra- and inter-individual distances
intra_distances = [];
inter_distances = [];

% Iterate through each unique label
unique_labels = unique(labels);
for i = 1:length(unique_labels)
    label = unique_labels{i};
    indices = find(strcmp(labels, label));

    % Compute intra-individual distances
    if numel(indices) > 1
        intra_distances = [intra_distances; reshape(pdist2(points(indices, :),points(indices,:)),[],1)];
    end

    % Compute inter-individual distances
    other_labels = unique_labels([1:i-1, i+1:end]);
    for j = 1:length(other_labels)
        other_label = other_labels{j};
        other_indices = find(strcmp(labels, other_label));
        inter_distances = [inter_distances; reshape(pdist2(points(indices, :), points(other_indices, :)), [], 1)];
    end
end

% Compute mean and standard deviation of the combined dataset
mean_distance_all = mean([intra_distances; inter_distances]);
std_distance_all = std([intra_distances; inter_distances]);

% Calculate z-scores for each dataset separately
z_scores_intra = (intra_distances - mean_distance_all) / std_distance_all;
z_scores_inter = (inter_distances - mean_distance_all) / std_distance_all;

% Plot the distribution of z-scores for intra-individual distances
figure;
histogram(z_scores_intra, 'Normalization', 'probability');
xlabel('Z-Scored Euclidean Distance');
ylabel('Probability');
hold on
histogram(z_scores_inter, 'Normalization', 'probability');
legend('Intra-individual', 'Inter-individual');
title('Distribution of Z-scored Intra-individual and Inter-individual Euclidean Distances');

% Perform Mann-Whitney U test
[p_value, h, stats] = ranksum(z_scores_intra, z_scores_inter);

% Display the results
fprintf('Mann-Whitney U Test Results:\n')
fprintf('p-value: %.9f\n', p_value)


%% Figure 7: Plot X, Y and Z R-squared values of linear fits of MDS coordinates with speed to confirm that a linear fit is appropriate. 
% Any p-values that are less than 0.05 then make the point plot an asterix
% otherwise a filled dot. 

linear_regress_r2_xstore = [];
linear_regress_r2_ystore = [];
linear_regress_r2_zstore = [];

linear_regress_p_xstore = [];
linear_regress_p_ystore = [];
linear_regress_p_zstore = [];

linear_regress_slope_xstore = [];
linear_regress_slope_ystore = [];
linear_regress_slope_zstore = [];

%filepath = '/Users/tanielwinner/Desktop/MDSSpeedRegression_individuals.pdf';

U = unique(SubjectTrainTrials);

%for each subject plot linear fit of MDS axis and speed
for s = 1:length(U)

    sub_idx = find(SubjectTrainTrials == U(s)); %get subject indices
    x = SpeedTrain(sub_idx)*0.01;
    y1 = Y3(sub_idx,1); %MDSx location
    y2 = Y3(sub_idx,2); %MDSy location
    y3 = Y3(sub_idx,3); %MDSz location

    colors = pointColors(sub_idx,:);

    % plot MDSx vs speed
    % figure(s)
    % set(gcf, 'Position', [100 100 200 100]);
    % sgtitle(U(s))
    % subplot(1,3,1)
    [r, p_value] = corrcoef(x, y1);
    coefficients = polyfit(x, y1, 1);
    x_values = linspace(min(x), max(x), 100);
    y_values = polyval(coefficients, x_values);
    r_value = r(1, 2)^2;
    linear_regress_r2_xstore = [linear_regress_r2_xstore;  r_value];
    p_value = p_value(1, 2);
    linear_regress_p_xstore = [linear_regress_p_xstore; p_value];

    linear_regress_slope_xstore = [linear_regress_slope_xstore; coefficients(1)];

    
    % % Plot the scatter plot
    % scatter(x, y1, 100, colors, 'filled');
    % xlabel('Speed (m/s)');
    % ylabel('MDS 1');
    % title('X')
    % 
    % hold on
    % % Plot the regression line
    % plot(x_values, y_values, 'r', 'LineWidth', 2);
    % 
    % text(0.1, 0.9, sprintf('Correlation Coefficient: %.2f\np-value: %.3e\n', r_value, p_value), ...
    %     'Units', 'normalized', 'FontSize', 15, 'FontWeight', 'bold');
    % %legend('Data', 'Linear Fit', 'Location', 'NorthWest');
    % hold on;

    % plot MDSy vs speed
    % subplot(1,3,2)
    [r, p_value] = corrcoef(x, y2);
    coefficients = polyfit(x, y2, 1);
    x_values = linspace(min(x), max(x), 100);
    y_values = polyval(coefficients, x_values);
    r_value = r(1, 2)^2;
    linear_regress_r2_ystore = [linear_regress_r2_ystore;  r_value];
    p_value = p_value(1, 2);
    linear_regress_p_ystore = [linear_regress_p_ystore; p_value];
    linear_regress_slope_ystore = [linear_regress_slope_ystore; coefficients(1)];



     % plot MDSz vs speed
    % subplot(1,3,3)
    [r, p_value] = corrcoef(x, y3);
    coefficients = polyfit(x, y3, 1);
    x_values = linspace(min(x), max(x), 100);
    y_values = polyval(coefficients, x_values);
    r_value = r(1, 2)^2;
    linear_regress_r2_zstore = [linear_regress_r2_zstore;  r_value];
    p_value = p_value(1, 2);
    linear_regress_p_zstore = [linear_regress_p_zstore; p_value];
    linear_regress_slope_zstore = [linear_regress_slope_zstore; coefficients(1)];
    


end

% Adjust the position for each model
box_positions = 1:3;
% Labels for the data types
jitterAmount = 0.1;
data_type_labels = {"X", "Y", "Z"};
separation_factor  = 0.2;

R2_xyz = [linear_regress_r2_xstore linear_regress_r2_ystore linear_regress_r2_zstore ];
P_xyz = [linear_regress_p_xstore linear_regress_p_ystore linear_regress_p_zstore]; 

f = find(trialtype == 1);
unique_col = pointColors(f,:);

figure()
for b = 1:3 % 
    
    r2 = R2_xyz(:,b);
    box_x = box_positions(b);
    adjusted_positions = box_positions(b) * separation_factor; % Adjust the position
    boxplot(r2, 'Positions', box_positions(b), 'Notch', 'on');
    hold on
    num_data_points = length(r2);
    x_jittered = box_x + (rand(1, num_data_points) - 0.5) * jitterAmount;
    
    for c = 1:length(r2);
        SubBoxColors = unique_col(c,:);

        if P_xyz(c,b) < 0.05 %get an idea of the non-statistical fits
            scatter(x_jittered(c), r2(c), 200, SubBoxColors,'filled');
            hold on
            xticks(box_positions);
            xticklabels(data_type_labels);
        else
            scatter(x_jittered(c), r2(c), 200, SubBoxColors,'o');
            hold on
            xticks(box_positions);
            xticklabels(data_type_labels);
        end

    end

end

xlabel('MDS Axes');
ylabel('Magnitude');
title(' Simple linear regression of individuals MDS coordinates vs. speed')


%% Plot X, Y and Z regression SLOPES of linear fits of MDS coordinates with speed to confirm that a linear fit is appropriate. 
% Any p-values that are less than 0.05 then make the point plot an asterix
% otherwise a filled dot. 

% Adjust the position for each model
box_positions = 1:3;
% Labels for the data types
jitterAmount = 0.1;
data_type_labels = {"X", "Y", "Z"};
separation_factor  = 0.2;

R2_xyz = [linear_regress_slope_xstore linear_regress_slope_ystore linear_regress_slope_zstore ];
P_xyz = [linear_regress_p_xstore linear_regress_p_ystore linear_regress_p_zstore]; 

f = find(trialtype == 1);
unique_col = pointColors(f,:);

figure()
for b = 1:3 % 
    
    r2 = R2_xyz(:,b);
    box_x = box_positions(b);
    adjusted_positions = box_positions(b) * separation_factor; % Adjust the position
    boxplot(r2, 'Positions', box_positions(b), 'Notch', 'on');
    hold on
    num_data_points = length(r2);
    x_jittered = box_x + (rand(1, num_data_points) - 0.5) * jitterAmount;
    
    for c = 1:length(r2);
        SubBoxColors = unique_col(c,:);

        if P_xyz(c,b) < 0.05 %get an idea of the non-statistical fits
            scatter(x_jittered(c), r2(c), 200, SubBoxColors,'filled');
            hold on
            xticks(box_positions);
            xticklabels(data_type_labels);
        else
            scatter(x_jittered(c), r2(c), 200, SubBoxColors,'o');
            hold on
            xticks(box_positions);
            xticklabels(data_type_labels);
        end

    end

end

xlabel('MDS Axes');
ylabel('Slope Magnitude');
title(' Simple linear regression of individuals MDS coordinates vs. speed Slope')

%% A single run of Mixed Linear Effects Model on MDS coordinates of all subjects, all speeds


% regenerate the 3D MDS plot

figure(1000)
scatter3(Y3(1:end, 1), Y3(1:end, 2), Y3(1:end, 3), markerSize, pointColors, 'filled','MarkerEdgeColor', 'k','HandleVisibility', 'off');

%labels = unique(SubjectTrainTrials);

hold on;

f = find(trialtype == 1);
colors_f = pointColors(f,:);

for i = 1:size(colors_f, 1)
    scatter3(nan, nan, nan, 50, colors_f(i, :), 'filled');
end

legend(legendlabels, 'Location', 'eastoutside');


% Set axis equal and enable grid
axis equal;
grid on;
grid(gca, 'minor');

% Label the axes
xlabel('MDS 1');
ylabel('MDS 2');
zlabel('MDS 3');

% Add a title
title('3D MDS Gait Signatures fit with lme');



points_store = [];

SubjectID = categorical(SubjectTrainTrials);
MDS_3D = Y3;

speed = SpeedTrain;

MDS_X = MDS_3D(:,1);
MDS_Y = MDS_3D(:,2);
MDS_Z = MDS_3D(:,3);

tbl = table(MDS_X,MDS_Y,MDS_Z, speed,SubjectID,'VariableNames', {'Xpos', 'Ypos', 'Zpos','Speed','SubjectID'});

% Fast_XYZ is the dependent variable (to be predicted), SS_XYZ and SS speed are predictor
% variables used to explain or predict Fast_XYZ. SubjectID is a random
% intercept for each SubjectID

lme_model_X = fitlme(tbl, 'Xpos ~ Speed + (1|SubjectID)');
lme_model_Y = fitlme(tbl, 'Ypos ~ Speed + (1|SubjectID)');
lme_model_Z = fitlme(tbl, 'Zpos ~ Speed + (1|SubjectID)');

% Example evaluation - Residual analysis
% residuals = lme_model_Z.Residuals.Raw;
% figure()
% plotResiduals(lme_model_Z, 'histogram');
% figure()
% plotResiduals(lme_model_Z, 'fitted');
% figure()
% qqplot(residuals);

% Example evaluation - R-squared and adjusted R-squared
% fprintf('R-squared: %.4f\n', lme_model_Z.Rsquared.Ordinary);
% fprintf('Adjusted R-squared: %.4f\n', lme_model_Z.Rsquared.Adjusted);

% Display model summaries
disp('Model summary for X position:');
disp(lme_model_X);
disp('Model summary for Y position:');
disp(lme_model_Y);
disp('Model summary for Z position:');
disp(lme_model_Z);

%
%Obtain p-values from the model
pX = anova(lme_model_X, 'DFMethod', 'Residual').pValue(2);
pY = anova(lme_model_Y, 'DFMethod', 'Residual').pValue(2);
pZ = anova(lme_model_Z, 'DFMethod', 'Residual').pValue(2);
% 
% % Display results
disp(['Significance test for X position: p-value = ', num2str(pX)]);
disp(['Significance test for Y position: p-value = ', num2str(pY)]);
disp(['Significance test for Z position: p-value = ', num2str(pZ)]);

% Plot vectors
% Extract fixed effects coefficients for X position
betaX = fixedEffects(lme_model_X);

% Extract random effects coefficients for X position
bX = randomEffects(lme_model_X);

% Extract fixed effects coefficients for Y position
betaY = fixedEffects(lme_model_Y);

% Extract random effects coefficients for Y position
bY = randomEffects(lme_model_Y);

% Extract fixed effects coefficients for Z position
betaZ = fixedEffects(lme_model_Z);

% Extract random effects coefficients for Z position
bZ = randomEffects(lme_model_Z);

% Assuming you have a vector of unique SubjectIDs
uniqueSubjectIDs = unique(SubjectID);

% Plot linear lines in 3D space for each subject
f = find(trialtype ==1);
SubColors = pointColors(f,:); % get a single color for each subject

for i = 1:length(uniqueSubjectIDs)
    subjectID = uniqueSubjectIDs(i);

    % Find indices for the current subject
    indices = (SubjectID == subjectID);

    % Extract speed values for the current subject
    speedValues = speed(indices);

    % Predict X, Y, and Z positions for the current subject
    predictedX = betaX(1) + betaX(2) * speedValues + bX(i);
    predictedY = betaY(1) + betaY(2) * speedValues + bY(i);
    predictedZ = betaZ(1) + betaZ(2) * speedValues + bZ(i);

    points = [predictedX,predictedY,predictedZ];
    points_store = [points_store; points];
    
    
    figure(1000);
    hold on

    %scatter3(points(:, 1), points(:, 2), points(:, 3), 'filled');
    %hold on;
    x = points(:, 1);
    y = points(:, 2);
    z = points(:, 3);
    plot3(x, y, z, '-', 'Color',SubColors(i,:), 'LineWidth', 10);
    % Draw arrow vectors connecting the points
% for j = 1:size(points, 1)-1
%     quiver3(points(j, 1), points(j, 2), points(j, 3), ...
%             points(j+1, 1) - points(j, 1), points(j+1, 2) - points(j, 2), points(j+1, 3) - points(j, 3), ...
%             'Color', SubColors(i,:), 'LineWidth', 20);
% 
%     hold on
% 
% end

end
% 
% hold off;
% 
% % Evaluate Model Predictions
% 
% % Assuming you have predictedPoints (150x3) and actualPoints (150x3)
% 
% % Step 1: Calculate Euclidean distances
% euclideanDistances = sqrt(sum((points_store - MDS_3D).^2, 2));
% 
% % Step 2: Calculate other statistical measures
% meanDistance = mean(euclideanDistances);
% medianDistance = median(euclideanDistances);
% minDistance = min(euclideanDistances);
% maxDistance = max(euclideanDistances);
% stdDevDistance = std(euclideanDistances);
% 
% % Display results
% disp(['Mean Euclidean Distance: ' num2str(meanDistance)]);
% disp(['Median Euclidean Distance: ' num2str(medianDistance)]);
% disp(['Minimum Euclidean Distance: ' num2str(minDistance)]);
% disp(['Maximum Euclidean Distance: ' num2str(maxDistance)]);
% disp(['Standard Deviation of Euclidean Distances: ' num2str(stdDevDistance)]);
% 
% % You can also visualize the distribution of distances if needed
% figure;
% histogram(euclideanDistances, 'BinWidth', 1);
% xlabel('Euclidean Distance');
% ylabel('Frequency');
% title('Distribution of Euclidean Distances between predicted and actual MDS coordinates');

%% Hierarchial Bootstrapping LME on Subject Trials and Speed Trials (4 to 7) 
figure(2000)
scatter3(Y3(1:end, 1), Y3(1:end, 2), Y3(1:end, 3), markerSize, pointColors, 'filled','MarkerEdgeColor', 'k','HandleVisibility', 'off');

%labels = unique(SubjectTrainTrials);

hold on;

f = find(trialtype == 1);
colors_f = pointColors(f,:);

for i = 1:size(colors_f, 1)
    scatter3(nan, nan, nan, 50, colors_f(i, :), 'filled');
end

legend(legendlabels, 'Location', 'eastoutside');


% Set axis equal and enable grid
axis equal;
grid on;
grid(gca, 'minor');

% Label the axes
xlabel('MDS 1');
ylabel('MDS 2');
zlabel('MDS 3');

% Add a title
title('3D MDS Gait Signatures fit with Hierarchial Bootstrapped lmes');

BetaX_fixed_store = [];
bX_random_store = [];
px_random_store = [];

BetaY_fixed_store = [];
bY_random_store = [];
pY_random_store = [];

BetaZ_fixed_store = [];
bZ_random_store = [];
pZ_random_store = [];

LOO_models_storeX = cell(17,1);
LOO_models_storeY = cell(17,1);
LOO_models_storeZ = cell(17,1);

pvalue_X_store = [];
pvalue_Y_store = [];
pvalue_Z_store = [];

held_out_sub = [];
speed_store = [];
sub_id_store = [];

num_bstrap = 6;
MDS_X = [];
MDS_Y = [];
MDS_Z = [];

pX = [];
pY = [];
pZ = [];

R2_X = [];
R2_Y = [];
R2_Z = [];

MDS_3D = [];
MDS_3D_store = [];

meanDistance = [];
stdDevDistance = [];
agg_samples = [];
Sub_store = [];
Color_store = [];
Trial_store = [];
Speed_store = [];
points_store = []; %predicted points with matching subject labels and color_store
euclideanDistances_store = [];

sub_unique = unique(SubjectTrainTrials); % get list of each subject

% in each model run - one subject is left out

for sub = 1:length(sub_unique) %17 LOO models
    display(['Subject LOO # : ', num2str(sub)])
    sub_idx = find(SubjectTrainTrials ~= sub_unique(sub)); %get subject indices that we keep
    held_out_sub = [held_out_sub; sub_unique(sub)]; %store which subject was held out on each model run

    % update new variables to exclude the data of the left out subject
    SubjectTrainTrials_exc = SubjectTrainTrials(sub_idx);
    speed_exc = SpeedTrain(sub_idx);
    Y3_exc = Y3(sub_idx,:);
    pointColors_exc = pointColors(sub_idx,:);
    trialtype_exc = trialtype(sub_idx);

    U = unique(SubjectTrainTrials_exc); %list of included subjects in model training

    for b = 1:num_bstrap % for each sub left in training - run bootstrap x times
        %display(['Bootstrap # : ', num2str(b)])
        for num_trials = 4:7  % alter the number of speed trials in the training from 4 to 7 speeds
            %display(['Num_trials # : ', num2str(num_trials)])
      
            MDS_X = [];
            MDS_Y = [];
            MDS_Z = [];
            speed = [];
            SubjectID = [];
            agg_samples = [];

            % aggregate samples per bootstrap run across subjects
            for u = 1:length(U) %for each subject in inclusive set
                sub_trial_indices = find(SubjectTrainTrials_exc == U(u)); % find all trial indices for this subject
                bootstrapped_sample_indices = datasample(sub_trial_indices, num_trials,'Replace',false); % sample x random indices each time/run with replacement (not within each run)
                % extract data with bootstrapped indices 
                agg_samples = [agg_samples; bootstrapped_sample_indices]; %all sample indices for all subjects in this bootstrap run

            end


            MDS_3D = Y3_exc(agg_samples,:); % get MDS locations for this run
            MDS_3D_store = [MDS_3D_store;MDS_3D];
            speed = speed_exc(agg_samples);
            MDS_X = MDS_3D(:,1);
            MDS_Y = MDS_3D(:,2);
            MDS_Z = MDS_3D(:,3);
            SubjectID = categorical(SubjectTrainTrials_exc(agg_samples));

            SubjectTrainTrials_exc_boot = SubjectTrainTrials_exc(agg_samples);
            pointColors_exc_boot = pointColors_exc(agg_samples,:);
            trialtype_exc_boot = trialtype_exc(agg_samples);

            %Sub_store = [Sub_store; SubjectTrainTrials_exc(agg_samples)];
            %Color_store = [Color_store; pointColors_exc(agg_samples,:)];
            %Trial_store = [Trial_store; trialtype_exc(agg_samples)];
            %Speed_store = [Speed_store; speed_exc(agg_samples)];

            % table of bootstrapped data
            tbl = table(MDS_X,MDS_Y,MDS_Z, speed,SubjectID,'VariableNames', {'Xpos', 'Ypos', 'Zpos','Speed','SubjectID'});

            % Fast_XYZ is the dependent variable (to be predicted), SS_XYZ and SS speed are predictor
            % variables used to explain or predict Fast_XYZ. SubjectID is a random
            % intercept for each SubjectID

            lme_model_X = fitlme(tbl, 'Xpos ~ Speed + (1|SubjectID)');
            lme_model_Y = fitlme(tbl, 'Ypos ~ Speed + (1|SubjectID)');
            lme_model_Z = fitlme(tbl, 'Zpos ~ Speed + (1|SubjectID)');

            R2_X = [R2_X; lme_model_X.Rsquared.Adjusted];
            R2_Y = [R2_Y; lme_model_Y.Rsquared.Adjusted];
            R2_Z = [R2_Z; lme_model_Z.Rsquared.Adjusted];


            % store models
            LOO_models_storeX{sub,1} = lme_model_X;
            LOO_models_storeY{sub,1} = lme_model_Y;
            LOO_models_storeZ{sub,1} = lme_model_Z;

            %Obtain p-values from the model
            pX = anova(lme_model_X, 'DFMethod', 'Residual').pValue(2);
            pY = anova(lme_model_Y, 'DFMethod', 'Residual').pValue(2);
            pZ = anova(lme_model_Z, 'DFMethod', 'Residual').pValue(2);

            % String to remove
            stringToRemove = sub_unique(sub);

            % Find the index of the string to remove
            indexToRemove = strcmp(sub_unique, stringToRemove);

            % Create a new variable with remaining strings
            train_sub = sub_unique(~indexToRemove);

            sub_id_store = [sub_id_store;train_sub];

            % Extract fixed effects coefficients for X position
            betaX = fixedEffects(lme_model_X)';

            % Extract random effects coefficients for X position
            bX = randomEffects(lme_model_X);

            % Extract fixed effects coefficients for Y position
            betaY = fixedEffects(lme_model_Y)';

            % Extract random effects coefficients for Y position
            bY = randomEffects(lme_model_Y);

            % Extract fixed effects coefficients for Z position
            betaZ = fixedEffects(lme_model_Z)';

            % Extract random effects coefficients for Z position
            bZ = randomEffects(lme_model_Z);

            % Assuming you have a vector of unique SubjectIDs
            uniqueSubjectIDs = unique(SubjectID);

            % update to store concatenate values from each model run.
            BetaX_fixed_store = [BetaX_fixed_store; betaX];
            bX_random_store = [bX_random_store; bX];
            px_random_store = [px_random_store; repelem(pX,16)'];

            BetaY_fixed_store = [BetaY_fixed_store; betaY];
            bY_random_store = [bY_random_store; bY];
            pY_random_store = [pY_random_store; repelem(pY,16)'];

            BetaZ_fixed_store = [BetaZ_fixed_store; betaZ];
            bZ_random_store = [bZ_random_store;bZ];
            pZ_random_store = [pZ_random_store; repelem(pZ,16)'];

            pvalue_X_store = [pvalue_X_store; pX]; %for comparing to model runs
            pvalue_Y_store = [pvalue_Y_store; pY];
            pvalue_Z_store = [pvalue_Z_store; pZ];


            f = find(trialtype_exc ==1);

            SubColors = pointColors_exc(f,:); % get a single color for each subject

            predicted_points_bootstrap = [];

            for i = 1:length(uniqueSubjectIDs)
                subjectID = uniqueSubjectIDs(i);

                % Find indices for the current subject
                indices = (SubjectID == subjectID);

                % Extract speed values for the current subject
                speedValues = speed(indices);


                % Predict X, Y, and Z positions for the current subject
                predictedX = betaX(1) + betaX(2) * speedValues + bX(i);
                predictedY = betaY(1) + betaY(2) * speedValues + bY(i);
                predictedZ = betaZ(1) + betaZ(2) * speedValues + bZ(i);

                points = [predictedX,predictedY,predictedZ];

                points_store = [points_store; points]; % predicted points
                Sub_store = [Sub_store; SubjectTrainTrials_exc_boot(indices)];
                Color_store = [Color_store; pointColors_exc_boot(indices,:)];
                Trial_store = [Trial_store; trialtype_exc_boot(indices)];
                Speed_store = [Speed_store;  speedValues];


                %scatter3(points(:, 1), points(:, 2), points(:, 3), 'filled');
                %hold on;
                x = points(:, 1);
                y = points(:, 2);
                z = points(:, 3);

  %%%%%%%%%%%%%%%%%%%%% UNCOMMENT BELOW TO PLOT ALL BOOTSTRAPPED FITS %%%%%%%%%%%%%%%% Takes some time
                
                % figure(2000) % specify the previously plotted 3D MDS plot
                % plot3(x, y, z, '-', 'Color',SubColors(i,:), 'LineWidth', 1);
                % hold on
                % Draw arrow vectors connecting the points

                %legend_handle = legend;    
               % set(legend_handle, 'Visible', 'off');
                
                predicted_points_bootstrap = [predicted_points_bootstrap; points]; % predicted points of bootstrap

            end

            %hold off;

            % Evaluate Model Predictions

            % Assuming you have predictedPoints (150x3) and actualPoints (150x3)

            % Step 1: Calculate Euclidean distances
            euclideanDistances = sqrt(sum((predicted_points_bootstrap - MDS_3D).^2, 2));
            euclideanDistances_store = [euclideanDistances_store; euclideanDistances]; %per point
            % Step 2: Calculate other statistical measures'

            meanDistance = [meanDistance; mean(euclideanDistances)]; %per bootstrap/model
            stdDevDistance = [stdDevDistance; std(euclideanDistances)];


        end

    end

end


predictedValues = points_store;
actualValues = MDS_3D_store;

% Assuming you have predictedValues (150x3) and actualValues (150x3)

% Step 1: Calculate residuals for each dimension
residualsX = actualValues(:, 1) - predictedValues(:, 1);
residualsY = actualValues(:, 2) - predictedValues(:, 2);
residualsZ = actualValues(:, 3) - predictedValues(:, 3);

% Step 2: Create a figure for residual vs. predicted values
figure;

% Plot residuals vs. predicted values for each dimension
subplot(3, 1, 1);
scatter(predictedValues(:, 1), residualsX);
xlabel('Predicted X');
ylabel('Residuals X');
title('Residuals vs. Predicted')
subplot(3, 1, 2);
scatter(predictedValues(:, 2), residualsY);
xlabel('Predicted Y');
ylabel('Residuals Y');
%title('Y');

subplot(3, 1, 3);
scatter(predictedValues(:, 3), residualsZ);
xlabel('Predicted Z');
ylabel('Residuals Z');
%title('Z');

% Create a figure for histogram plots with distribution curves
figure;

% Plot histograms of residuals for each dimension
subplot(3, 1, 1);
histogram(residualsX, 'BinWidth', 1, 'Normalization', 'probability');
xlabel('Residuals X');
ylabel('Probability');
%title('X')
title('Histogram of Residuals');

% Add distribution curve
hold on;
x_values = linspace(min(residualsX), max(residualsX), 100);
pdf_values = pdf(fitdist(residualsX, 'Normal'), x_values);
plot(x_values, pdf_values, 'LineWidth', 2, 'Color', 'r');
legend('Histogram', 'Distribution Curve');
hold off;

subplot(3, 1, 2);
histogram(residualsY, 'BinWidth', 1, 'Normalization', 'probability');
xlabel('Residuals Y');
ylabel('Probability');
%title('Y');

% Add distribution curve
hold on;
x_values = linspace(min(residualsY), max(residualsY), 100);
pdf_values = pdf(fitdist(residualsY, 'Normal'), x_values);
plot(x_values, pdf_values, 'LineWidth', 2, 'Color', 'r');
legend('Histogram', 'Distribution Curve');
hold off;

subplot(3, 1, 3);
histogram(residualsZ, 'BinWidth', 1, 'Normalization', 'probability');
xlabel('Residuals Z');
ylabel('Probability');
%title('Z');

% Add distribution curve
hold on;
x_values = linspace(min(residualsZ), max(residualsZ), 100);
pdf_values = pdf(fitdist(residualsZ, 'Normal'), x_values);
plot(x_values, pdf_values, 'LineWidth', 2, 'Color', 'r');
legend('Histogram', 'Distribution Curve');
hold off;

% Display the mean and standard deviation of residuals for each dimension
meanResidualX = mean(residualsX);
stdDevResidualX = std(residualsX);
disp(['Mean Residual X: ' num2str(meanResidualX)]);
disp(['Standard Deviation of Residuals X: ' num2str(stdDevResidualX)]);

meanResidualY = mean(residualsY);
stdDevResidualY = std(residualsY);
disp(['Mean Residual Y: ' num2str(meanResidualY)]);
disp(['Standard Deviation of Residuals Y: ' num2str(stdDevResidualY)]);

meanResidualZ = mean(residualsZ);
stdDevResidualZ = std(residualsZ);
disp(['Mean Residual Z: ' num2str(meanResidualZ)]);
disp(['Standard Deviation of Residuals Z: ' num2str(stdDevResidualZ)]);

%% Figure 8: Balance beam correlations 
% Get Euclidean Distances between Extreme Fast and Extreme Slow and SS Sigs for Gait Sigs

custom_colors =  [
    0, 0, 1;     % Blue - change?! same as Royal Blue
    1, 0, 0;     % Red
    0, 1, 1;     % Cyan
    1, 0, 1;     % Magenta
    0.5, 0, 0;   % Maroon
    1, 1, 0;     % Yellow
    0, 0, 0;     % Black
     0, 0, 1;    % Royal Blue
    0, 0, 0.5;   % Navy
    1, 0.5, 0;   % Orange
     0.3,0,0.5;   % DarkIndigo  
    0, 0.5, 0;   % Dark Green
    0, 0.5, 1;   % Light Blue
    1, 0, 0.5;   % Dark Pink
    0.5, 0.5, 1; % Light Indigo
    0.8, 0.6, 0.9; % Light Purple
    0.5, 0.5, 0; % Olive Green
    1, 0.5, 0.5; % Light Pink
    0.9, 0.9, 0.6; % Khaki
    0.5, 1, 0.5; % Light Green
];

% SET WHICH SIGNATURE DATA TYPE YOU WANT TO RUN
GaitSigs = Signatures.GS_3Dkin{1,1};
speed = SpeedTrain;

% Calculate distance to each trials' slowest fixed speed
SlowestSpeed_Sig = [];
SS_Speed_Sig = [];
W2R_Speed_Sig = [];
Speed_W2R = [];
Slow_Fast = [];
SpeedChange_SlowFast  = [];
SpeedChange_SSSlow = [];
SpeedChange_SSW2R = [];

trial_log_index_slow = trialtype == 1;  %find index of all slow (0.3m/s) speed condition
trial_log_index_w2r = trialtype == 8;%8; % find index of w2r transition speed --> try just fast
trial_log_index_sstm = trialtype == 3; % speed of SS TM

U = unique(SubjectTrainTrials);

for g = 1:length(U) %for each subject
    sp_log_index = ismember(SubjectTrainTrials,U(g));  % find the subject  
    slow_speed_idx = find(sp_log_index & trial_log_index_slow); %find slowest speed index of that subject
    w2r_speed_idx = find(sp_log_index & trial_log_index_w2r); %find w2r speed index of that subject
    sstm_speed_idx = find(sp_log_index & trial_log_index_sstm); % find ss_tm

    slow_speed_sig = GaitSigs(slow_speed_idx,:); 
    w2r_speed_sig = GaitSigs(w2r_speed_idx,:); 
    sstm_speed_sig = GaitSigs(sstm_speed_idx,:);


    speed = SpeedTrain;
    SpeedChange_SlowFast = [SpeedChange_SlowFast; (speed(w2r_speed_idx) - speed(slow_speed_idx))*0.01];
    SpeedChange_SSSlow = [SpeedChange_SSSlow;(speed(sstm_speed_idx) - speed(slow_speed_idx))*0.01];
    SpeedChange_SSW2R = [SpeedChange_SSW2R;(speed(w2r_speed_idx) - speed(sstm_speed_idx))*0.01];

    Speed_W2R = [Speed_W2R; speed(w2r_speed_idx)];
    SlowestSpeed_Sig = [SlowestSpeed_Sig;slow_speed_sig];
    W2R_Speed_Sig = [W2R_Speed_Sig;w2r_speed_sig];
    SS_Speed_Sig = [SS_Speed_Sig;sstm_speed_sig];
end



SS_Slow  = sqrt(sum((SlowestSpeed_Sig'-SS_Speed_Sig').^2))';
SS_Fast = sqrt(sum((W2R_Speed_Sig'-SS_Speed_Sig').^2))';
Slow_Fast = sqrt(sum((W2R_Speed_Sig'-SlowestSpeed_Sig').^2))';
SubjectID = U;


pointColors = custom_colors(sub_num, :); 

f = find(trialtype ==1); %find all fixed slow speed indices (everyone has this trial) so we get each subject color

SubColors = pointColors(f,:); % get a single color for each subject

% Join Tables

EucDistTable = table(SubjectID, SS_Slow,SS_Fast, Slow_Fast,SubColors,SpeedChange_SlowFast,SpeedChange_SSSlow,SpeedChange_SSW2R);


FullTable_WithDist = join(HYA_Study_InformationTable, EucDistTable,'Keys','SubjectID');

%% Distance between SS and fixed slow signatures and balance beam


f = find(trialtype == 3); %get one instance per subject
colors = [FullTable_WithDist.SubColors(f,1) FullTable_WithDist.SubColors(f,2) FullTable_WithDist.SubColors(f,3)];


x = FullTable_WithDist.SS_Slow(f); % change here depending on what is being correlated
y = FullTable_WithDist.BeamBalance(f);
[r, p_value] = corrcoef(x, y);
coefficients = polyfit(x, y, 1);
x_values = linspace(min(x), max(x), 100);
y_values = polyval(coefficients, x_values);
r_value = r(1, 2);
p_value = p_value(1, 2);

% Plot the scatter plot
figure();
scatter(x, y, 100, colors, 'filled');
xlabel('Distance between SS and extreme slow signatures'); % change here depending on what is being correlated
ylabel('Balance Beam Score (ft)');
hold on;

% Plot the regression line
plot(x_values, y_values, 'r', 'LineWidth', 2);

text(0.5, 0.9, sprintf('Correlation Coefficient: %.2f\np-value: %.2e', r_value, p_value), ...
    'Units', 'normalized', 'FontSize', 15, 'FontWeight', 'bold');
legend('Data', 'Linear Fit', 'Location', 'NorthWest');



%% Correlation between SS and extreme slow signature distance change vs. SS speed

f = find(trialtype == 3); %get one instance per subject
colors = [FullTable_WithDist.SubColors(f,1) FullTable_WithDist.SubColors(f,2) FullTable_WithDist.SubColors(f,3)];

x = cell2mat(SpeedTrainTrials(f))*0.01;
%x = FullTable_WithDist.SpeedChange_SSSlow(f); % change here depending on what is being correlated
y = FullTable_WithDist.SS_Slow(f); % change here depending on what is being correlated
[r, p_value] = corrcoef(x, y);
coefficients = polyfit(x, y, 1);
x_values = linspace(min(x), max(x), 100);
y_values = polyval(coefficients, x_values);
r_value = r(1, 2);
p_value = p_value(1, 2);

% Plot the scatter plot
figure();
scatter(x, y, 100, colors, 'filled');
xlabel('Speed change (m/s)');
ylabel('Distance between SS and extreme slow signatures'); % change here depending on what is being correlated
hold on;

% Plot the regression line
plot(x_values, y_values, 'r', 'LineWidth', 2);

text(0.5, 0.9, sprintf('Correlation Coefficient: %.2f\np-value: %.2e', r_value, p_value), ...
    'Units', 'normalized', 'FontSize', 15, 'FontWeight', 'bold');
legend('Data', 'Linear Fit', 'Location', 'NorthWest');


%% Correlation between Balance vs. SS speed 

f = find(trialtype == 3); %get one instance per subject
colors = [FullTable_WithDist.SubColors(f,1) FullTable_WithDist.SubColors(f,2) FullTable_WithDist.SubColors(f,3)];


x = cell2mat(SpeedTrainTrials(f))*0.01;
y = FullTable_WithDist.BeamBalance(f);
[r, p_value] = corrcoef(x, y);
coefficients = polyfit(x, y, 1);
x_values = linspace(min(x), max(x), 100);
y_values = polyval(coefficients, x_values);
r_value = r(1, 2);
p_value = p_value(1, 2);

% Plot the scatter plot
figure();
scatter(x, y, 100, colors, 'filled');
xlabel('SS Speed (m/s)');
ylabel('Balance Beam (ft)');
hold on;

% Plot the regression line
plot(x_values, y_values, 'r', 'LineWidth', 2);

text(0.5, 0.9, sprintf('Correlation Coefficient: %.2f\np-value: %.2e', r_value, p_value), ...
    'Units', 'normalized', 'FontSize', 15, 'FontWeight', 'bold');
legend('Data', 'Linear Fit', 'Location', 'NorthWest');


%% MDS colored by Beam score


GaitSigs = Signatures.GS_3Dkin{1,1};

BalanceBeamScore = FullTable_WithDist.BeamBalance;


All_GS = [GaitSigs];
all_speed = [SpeedTrain];  % Append 0 speed for the centroid
speed = SpeedTrain;


% Create a dissimilarity matrix
D = pdist2(All_GS, All_GS, 'euclidean'); % Dissimilarity matrix
Y3 = mdscale(D, 3); % Get the 3D coordinates for plotting

% Define marker size range
minSize = 75;
maxSize =700;

% Calculate marker size based on speed
markerSize = minSize + (speed - min(speed)) .* ((maxSize - minSize) / (max(speed) - min(speed)));


% Scatter plot of points with size and color according to speed 
figure()
scatter3(Y3(1:end, 1), Y3(1:end, 2), Y3(1:end, 3), markerSize, BalanceBeamScore, 'filled','MarkerEdgeColor', 'k','HandleVisibility', 'off');
colorbar
%clim([0, 30])
%labels = unique(SubjectTrainTrials);

hold on;

% Set axis equal and enable grid
axis equal;
grid on;
grid(gca, 'minor');

% Label the axes
xlabel('MDS 1');
ylabel('MDS 2');
zlabel('MDS 3');

% Add a title
title('Gait Signatures of Healthy Young Adults Colored by Balance Beam Score');

%% Figure 9: All other correlations

%% Run Function for correlations
AllCorrelations(FullTable_WithDist);

%% Need to get the CHANGE IN TASK VARIABLE from SS for each subject

% for columns 4 to 13 of table: 'FullTable_WithDist' for each subject, extract the SS and extreme fast and
% slow value and subtract to find the difference and save. 

labels = FullTable_WithDist.Properties.VariableNames([1,4:13]);
taskvarchange_fast = cell(10,1);
taskvarchange_slow = cell(10,1);
sub_unique = unique(SubjectTrainTrials);

slowvar = [];
fastvar = [];

ChangeTableFast = table();
ChangeTableSlow = table();

ChangeTableFast.(labels{1}) = sub_unique;
ChangeTableSlow.(labels{1}) = sub_unique;

for k = 4:13 % for each feature
    for sub = 1:length(sub_unique) % identify each subject        
        SS_TM_idx = find(SubjectTrainTrials == sub_unique(sub) & trialtype == 3); %get SS index
        VFast_idx = find(SubjectTrainTrials == sub_unique(sub) & trialtype == 8); % get fastest speed index  
        VSlow_idx = find(SubjectTrainTrials == sub_unique(sub) & trialtype == 1); % get fastest speed index  
        
        slowvar = [slowvar; FullTable_WithDist{SS_TM_idx,k}{1,1}-FullTable_WithDist{VSlow_idx,k}{1,1}]; 
        fastvar = [fastvar; FullTable_WithDist{SS_TM_idx,k}{1,1}-FullTable_WithDist{VFast_idx,k}{1,1}]; 
        
    end
    ChangeTableFast.(labels{k-2}) = fastvar;
    ChangeTableSlow.(labels{k-2}) = slowvar;
    slowvar = [];
    fastvar = [];
end

ChangeTableFast.DistFast = SS_Fast;
ChangeTableSlow.DistSlow = SS_Slow;


%% Plot change in task variables against gait sig change between fast and SS
names = ChangeTableFast.Properties.VariableNames;
for i = 2:11   %column variables of interest
    name = names{i};
    y = ChangeTableFast{:, i};
    x = ChangeTableFast{:,12};
    corrPlot(x, y, name, SubColors)
end

%% Plot change in task variables against gait sig change between SLOW and SS
names = ChangeTableSlow.Properties.VariableNames;
for i = 2:11   %column variables of interest
    name = names{i};
    y = ChangeTableSlow{:, i};
    x = ChangeTableSlow{:,12};
    corrPlot(x, y, name, SubColors)
end

%% Speed vs. Cadence etc.
x = cell2mat(FullTable_WithDist{:,2})*0.01; %speed
colors = [FullTable_WithDist.SubColors(:,1) FullTable_WithDist.SubColors(:,2) FullTable_WithDist.SubColors(:,3)];
xlabel('Speed (m/s)')

for i = 4:13;
    y =  cell2mat(FullTable_WithDist{:,i});
    name =  FullTable_WithDist.Properties.VariableNames{i};
    
    [r, p_value] = corrcoef(x, y);
    coefficients = polyfit(x, y, 1);
    x_values = linspace(min(x), max(x), 100);
    y_values = polyval(coefficients, x_values);
    r_value = r(1, 2);
    p_value = p_value(1, 2);
    
    % Plot the scatter plot
    figure();
    scatter(x, y, 100, colors, 'filled');
    xlabel('Speed (m/s)');
    ylabel(name);
    hold on;
    
    % Plot the regression line
    plot(x_values, y_values, 'r', 'LineWidth', 2);
    
    text(0.5, 0.9, sprintf('Correlation Coefficient: %.2f\np-value: %.1e', r_value, p_value), ...
        'Units', 'normalized', 'FontSize', 15, 'FontWeight', 'bold');
    legend('Data', 'Linear Fit', 'Location', 'NorthWest');
    
end

%% Supplementary: Classification SVM of 26 Discrete Vars

% Remove these subjects --> after removal subjects (n = 14)
YA004_indices = find(SubjectTrainTrials == 'YA004');
YA015_indices = find(SubjectTrainTrials == 'YA015');
YA006_indices = find(SubjectTrainTrials == 'YA006');

remove_indices = [YA004_indices;YA015_indices;YA006_indices];

% Create a logical index for elements to keep
keep_indices = true(size(SubjectTrainTrials));
keep_indices(remove_indices) = false;
num_runs = 10; 
num_trials_per_sub = 9;
num_subjects = 14;
num_training_trials = 4;
average_accuracy = [];
std_accuracy = [];


GaitSigs = DiscreteVars.YA_Discrete26Variables_mat;
Altered_SubjLabels = SubjectTrainTrials(keep_indices);
Altered_GaitSigs = GaitSigs(keep_indices,:);
Altered_sub_num = sub_num(keep_indices);

accuracy_results = [];

% Create a template for a linear SVM classifier
template = templateSVM('Standardize', 1); % Standardize features

for run = 1:num_runs
    train_data = [];
    train_labels = [];
    test_data = [];
    test_labels = [];


    %rng(run); %set a random seed for reproducibility

    subs = unique(Altered_sub_num);

    for subject = 1:num_subjects
        rng(run); % Set a random seed for reproducibility each run
        shuffled_ind = randperm(num_trials_per_sub); %randomizes which speed trials each subject and run

        train_indices = shuffled_ind(1:num_training_trials);
        test_indices = shuffled_ind(num_training_trials + 1:num_trials_per_sub);

        sub_ind = find(Altered_sub_num == subs(subject)==1);

        train_trial_indices = sub_ind(train_indices);
        test_trial_indices = sub_ind(test_indices);

        test_data = [test_data;Altered_GaitSigs(test_trial_indices, :)]; %concatenate each subject's test data
        test_labels = [test_labels;Altered_sub_num(test_trial_indices)];

        train_data = [train_data;Altered_GaitSigs(train_trial_indices, :)]; %concatenate each subject's test data
        train_labels = [train_labels;Altered_sub_num(train_trial_indices)];

        % Train an SVM classifier on the training data
        svm_model = fitcecoc(train_data, train_labels, 'Learners', template, 'Coding', 'onevsall');
        % Predict labels for the test data
        predicted_labels = predict(svm_model, test_data);

        % Calculate accuracy for this fold
        accuracy = sum(predicted_labels == test_labels) / numel(test_labels);
        accuracy_results =  [accuracy_results; accuracy];

    end
        

end

average_accuracy = [average_accuracy mean(accuracy_results(:))];
std_accuracy = [std_accuracy std(accuracy_results(:))];


% Construct the confusion matrix
conf_matrix = confusionmat(test_labels, predicted_labels);
labels = unique(test_labels);

% Plot the confusion matrix
figure;
heatmap(labels, labels, conf_matrix, 'Colormap', jet, 'ColorbarVisible', 'on', 'FontSize', 8, 'XLabel', 'Predicted Labels', 'YLabel', 'True Labels', 'Title', 'Confusion Matrix');

%% Supplementary: Classification SVM of ONLY kinematic discrete vars (9 total, 18 bilateral)


Kinematic_variables = cell2mat(table2cell(YA_Discrete26Variables_table(:,[10:13,16:29])));

% Remove these subjects --> after removal subjects (n = 14)
YA004_indices = find(SubjectTrainTrials == 'YA004');
YA015_indices = find(SubjectTrainTrials == 'YA015');
YA006_indices = find(SubjectTrainTrials == 'YA006');

remove_indices = [YA004_indices;YA015_indices;YA006_indices];

% Create a logical index for elements to keep
keep_indices = true(size(SubjectTrainTrials));
keep_indices(remove_indices) = false;
num_runs = 10; 
num_trials_per_sub = 9;
num_subjects = 14;
num_training_trials = 4;
average_accuracy = [];
std_accuracy = [];


GaitSigs = Kinematic_variables;
Altered_SubjLabels = SubjectTrainTrials(keep_indices);
Altered_GaitSigs = GaitSigs(keep_indices,:);
Altered_sub_num = sub_num(keep_indices);

accuracy_results = [];

% Create a template for a linear SVM classifier
template = templateSVM('Standardize', 1); % Standardize features

for run = 1:num_runs
    train_data = [];
    train_labels = [];
    test_data = [];
    test_labels = [];


    %rng(run); %set a random seed for reproducibility

    subs = unique(Altered_sub_num);

    for subject = 1:num_subjects
        rng(run); % Set a random seed for reproducibility each run
        shuffled_ind = randperm(num_trials_per_sub); %randomizes which speed trials each subject and run

        train_indices = shuffled_ind(1:num_training_trials);
        test_indices = shuffled_ind(num_training_trials + 1:num_trials_per_sub);

        sub_ind = find(Altered_sub_num == subs(subject)==1);

        train_trial_indices = sub_ind(train_indices);
        test_trial_indices = sub_ind(test_indices);

        test_data = [test_data;Altered_GaitSigs(test_trial_indices, :)]; %concatenate each subject's test data
        test_labels = [test_labels;Altered_sub_num(test_trial_indices)];

        train_data = [train_data;Altered_GaitSigs(train_trial_indices, :)]; %concatenate each subject's test data
        train_labels = [train_labels;Altered_sub_num(train_trial_indices)];

        % Train an SVM classifier on the training data
        svm_model = fitcecoc(train_data, train_labels, 'Learners', template, 'Coding', 'onevsall');
        % Predict labels for the test data
        predicted_labels = predict(svm_model, test_data);

        % Calculate accuracy for this fold
        accuracy = sum(predicted_labels == test_labels) / numel(test_labels);
        accuracy_results =  [accuracy_results; accuracy];

    end
        

end

acc_kinematics_dist = accuracy_results;

average_accuracy = [average_accuracy mean(accuracy_results(:))];
std_accuracy = [std_accuracy std(accuracy_results(:))];


% Construct the confusion matrix
conf_matrix = confusionmat(test_labels, predicted_labels);
labels = unique(test_labels);

% Plot the confusion matrix
figure;
heatmap(labels, labels, conf_matrix, 'Colormap', jet, 'ColorbarVisible', 'on', 'FontSize', 8, 'XLabel', 'Predicted Labels', 'YLabel', 'True Labels', 'Title', 'Confusion Matrix');


%% Supplementary: Classification SVM of ONLY kinetic discrete vars (4 total, 8 bilateral)


Kinetic_variables = cell2mat(table2cell(YA_Discrete26Variables_table(:,[4:9,14:15])));

% Remove these subjects --> after removal subjects (n = 14)
YA004_indices = find(SubjectTrainTrials == 'YA004');
YA015_indices = find(SubjectTrainTrials == 'YA015');
YA006_indices = find(SubjectTrainTrials == 'YA006');

remove_indices = [YA004_indices;YA015_indices;YA006_indices];

% Create a logical index for elements to keep
keep_indices = true(size(SubjectTrainTrials));
keep_indices(remove_indices) = false;
num_runs = 10; 
num_trials_per_sub = 9;
num_subjects = 14;
num_training_trials = 4;


GaitSigs = Kinetic_variables;
Altered_SubjLabels = SubjectTrainTrials(keep_indices);
Altered_GaitSigs = GaitSigs(keep_indices,:);
Altered_sub_num = sub_num(keep_indices);

accuracy_results = [];

% Create a template for a linear SVM classifier
template = templateSVM('Standardize', 1); % Standardize features

for run = 1:num_runs
    train_data = [];
    train_labels = [];
    test_data = [];
    test_labels = [];


    %rng(run); %set a random seed for reproducibility

    subs = unique(Altered_sub_num);

    for subject = 1:num_subjects
        rng(run); % Set a random seed for reproducibility each run
        shuffled_ind = randperm(num_trials_per_sub); %randomizes which speed trials each subject and run

        train_indices = shuffled_ind(1:num_training_trials);
        test_indices = shuffled_ind(num_training_trials + 1:num_trials_per_sub);

        sub_ind = find(Altered_sub_num == subs(subject)==1);

        train_trial_indices = sub_ind(train_indices);
        test_trial_indices = sub_ind(test_indices);

        test_data = [test_data;Altered_GaitSigs(test_trial_indices, :)]; %concatenate each subject's test data
        test_labels = [test_labels;Altered_sub_num(test_trial_indices)];

        train_data = [train_data;Altered_GaitSigs(train_trial_indices, :)]; %concatenate each subject's test data
        train_labels = [train_labels;Altered_sub_num(train_trial_indices)];

        % Train an SVM classifier on the training data
        svm_model = fitcecoc(train_data, train_labels, 'Learners', template, 'Coding', 'onevsall');
        % Predict labels for the test data
        predicted_labels = predict(svm_model, test_data);

        % Calculate accuracy for this fold
        accuracy = sum(predicted_labels == test_labels) / numel(test_labels);
        accuracy_results =  [accuracy_results; accuracy];

    end
        

end

acc_kinetics_dist = accuracy_results;
average_accuracy_kinetics = mean(accuracy_results(:));
std_accuracy_kinetics = std(accuracy_results(:));


% Construct the confusion matrix
conf_matrix = confusionmat(test_labels, predicted_labels);
labels = unique(test_labels);

% Plot the confusion matrix
figure;
heatmap(labels, labels, conf_matrix, 'Colormap', jet, 'ColorbarVisible', 'on', 'FontSize', 8, 'XLabel', 'Predicted Labels', 'YLabel', 'True Labels', 'Title', 'Confusion Matrix');

%% Test differences in mean


% Perform Mann-Whitney U test
[p_value, h, stats] = ranksum(acc_kinematics_dist, acc_kinetics_dist);

% Display the results
fprintf('Mann-Whitney U Test Results:\n')
fprintf('p-value: %.9f\n', p_value)
