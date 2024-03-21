function [] = AllModels_GaitSignatureVisulizations(Signatures,ModelsToPlot, GaitPhaseTable)
%This function inputs all model data and plots 4 gait signatures visualizations for each of
%the specified model types in the 'ModelsToPlot' variable. 

load('/Users/tanielwinner/Desktop/eLIFE_MATLAB_092221/YA_study/Reprocessed_HYA_092223/matfile extracted data and labels/subjects_label.mat');
load('/Users/tanielwinner/Desktop/eLIFE_MATLAB_092221/YA_study/Reprocessed_HYA_092223/matfile extracted data and labels/sub_num_label.mat');
load('/Users/tanielwinner/Desktop/eLIFE_MATLAB_092221/YA_study/Reprocessed_HYA_092223/matfile extracted data and labels/speeds_label.mat');
load('/Users/tanielwinner/Desktop/eLIFE_MATLAB_092221/YA_study/Reprocessed_HYA_092223/matfile extracted data and labels/trialtype.mat');


modeltypes = {'GS 2D', 'GS 3D', 'GS kinetics','GS all','PC 2D', 'PC 3D', 'PC kinetics','PC all'};

for i = 1:length(ModelsToPlot); %number of feature models
    idx = ModelsToPlot(i);
    GaitSigs = Signatures{1,i}{1,1};
    condname = modeltypes(idx);
    gaitphase = GaitPhaseTable{1,i}{1,1};
 
    %% Plot 3D loops according to individual - PC 1,2,3
    
    % Able-bodied
    trialnum = size(GaitSigs,1); % number of trials
    GaitType = repelem("AB",trialnum); 
    conditions = SubjectTrainTrials;
    gaitgroup = GaitType;
    spec = 'AB'; % specify gait group
    pallette = "custom_colors";
    Plot3DConditions(GaitSigs,condname,gaitgroup,spec,conditions,pallette)
      
    %% Plot 3D loops according to individual - PC 4,5,6
    
    % Able-bodied
    GaitType = repelem("AB",trialnum); 
    conditions = SubjectTrainTrials;
    gaitgroup = GaitType;
    spec = 'AB'; % specify gait group
    pallette = "custom_colors";
    Plot3DConditions_PC456(GaitSigs,condname,gaitgroup,spec,conditions,pallette)
    
    
    %% Plot 3D loops according to speed
    % Able-bodied
    
    conditions =  SpeedTrain*0.01;
    gaitgroup = GaitType;
    spec = 'AB'; % specify gait group
    Plot3DSpeed(GaitSigs,condname,gaitgroup,spec,conditions)
    
    %% Plot 3D loops according to speed PC456
    % Able-bodied
    
    conditions =  SpeedTrain*0.01;
    gaitgroup = GaitType;
    spec = 'AB'; % specify gait group
    Plot3DSpeed456(GaitSigs,condname,gaitgroup,spec,conditions)

    %% Plot 3D loops according to gait phase
    gaitgroupindices = 1:trialnum;
    Plot3DGaitPhase(gaitgroupindices, GaitSigs, gaitphase,condname)

end