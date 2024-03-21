function [] = AllModels_SpeedLinear_Classification(Signatures)
% Random Train-Test Split Classification: Train linear classifier where 3 random speeds from each subject is selected as test set and 6 are used for training
% Total of 10 runs, rng set on each run to keep same across model runs
% since some individuals do not have all 9 speed conditions - we will leave out 3


load('/Users/tanielwinner/Desktop/eLIFE_MATLAB_092221/YA_study/Reprocessed_HYA_092223/matfile extracted data and labels/subjects_label.mat');
load('/Users/tanielwinner/Desktop/eLIFE_MATLAB_092221/YA_study/Reprocessed_HYA_092223/matfile extracted data and labels/sub_num_label.mat');
load('/Users/tanielwinner/Desktop/eLIFE_MATLAB_092221/YA_study/Reprocessed_HYA_092223/matfile extracted data and labels/speeds_label.mat');
load('/Users/tanielwinner/Desktop/eLIFE_MATLAB_092221/YA_study/Reprocessed_HYA_092223/matfile extracted data and labels/trialtype.mat');


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
num_training_trials = 6;
% average_accuracy = [];
% std_accuracy = [];

labels = {'GS 2Dkin', 'GS 3Dkin', 'GS kinetics', 'GS All Data', 'PC 2Dkin','PC 3Dkin','PC kinetics','PC All Data'}; 

num_runs = 10;
MSE_results = zeros(num_runs, 1);
svr_models = cell(num_runs, 1);
regression_coefficients = [];

% Specify the ratio for training and testing data
train_ratio = 6/9;
test_ratio = 3/9;

% % Polynomial kernel parameters
% poly_degree = 3; % Polynomial degree (can be adjusted)
% boxConstraint = 1; % Box constraint (can be adjusted)

modelpredictions = cell(8,1);
modelr2 = cell(8,1);
groundtruth_speed = [];
predicted_speed = [];
Rsq_consolidate = [];

% Define the number of trees for Random Forest
numTrees = 100;


for m = 1:8; % # number of models 

    GaitSigs = Signatures{1,m}{1,1};
    Altered_SubjLabels = SubjectTrainTrials(keep_indices);
    Altered_GaitSigs = GaitSigs(keep_indices,:);
    Altered_sub_num = sub_num(keep_indices);
    Altered_speed = SpeedTrain(keep_indices)*0.01; %m/s

for run = 1:num_runs
    % Set a random seed for reproducibility
    rng(run);
    
    % Split the data into training and testing sets
    num_samples = size(Altered_GaitSigs, 1);
    num_train_samples = round(train_ratio * num_samples);
    
    % Randomly shuffle the indices to create random training and testing sets
    shuffled_indices = randperm(num_samples);
    train_indices = shuffled_indices(1:num_train_samples);
    test_indices = shuffled_indices(num_train_samples+1:end);
    
    % Split the data and labels
    train_data = Altered_GaitSigs(train_indices, :);
    test_data = Altered_GaitSigs(test_indices, :);
    
    train_speed = Altered_speed(train_indices);
    test_speed = Altered_speed(test_indices);

    

    % Train a Random Forest model
    randomForestModel = fitensemble(train_data, train_speed, 'Bag', numTrees, 'Tree', 'Type', 'regression');

    predicted_speed_rf = predict(randomForestModel, test_data);
  
    % % Train a Gradient Boosting Machines (GBM) model
    % gbmModel = fitensemble(X', Y, 'LSBoost', 100, 'Tree', 'Type', 'regression');

    % Predict speed on the test data
    groundtruth_speed = [groundtruth_speed; test_speed];
    predicted_speed = [predicted_speed; predicted_speed_rf];
    % regression_coefficients = [regression_coefficients; coefficients(2)];
    % Rsq_consolidate = [Rsq_consolidate;Rsq];
end

   model_grdtruthspeeds{m,1} = groundtruth_speed;
   modelpredictions{m,1} = predicted_speed;
   modelregresscoeff{m,1} = regression_coefficients;
   modelr2{m,1} = Rsq_consolidate;
   n = length(predicted_speed); %number of model runs
   rmse_rf = sqrt(immse(predicted_speed, groundtruth_speed)); % Replace actual_speed with your true values
   
   figure()
   scatter(groundtruth_speed,predicted_speed);
   xlabel('Actual Speed (m/s)')
   ylabel('Predicted Speed (m/s)')
   title(labels{m})
   % Adding a diagonal line (y = x) for reference
   hold on;
   xline = linspace(min(groundtruth_speed), max(groundtruth_speed), 100);
   plot(xline, xline, 'r--'); % Red dashed line for y = x
   legend('Data Points', 'y = x', 'Location', 'southeast'); % Add legend
   text(0.5, 0.9, sprintf('RMSE: %.2f\n n = %d', rmse_rf, n), ...
   'Units', 'normalized', 'FontSize', 15, 'FontWeight', 'bold');
   legend('Data', 'y=x', 'Location', 'NorthWest');
   hold off;

    %reset variables
    groundtruth_speed = [];
    predicted_speed = [];
    regression_coefficients = [];
    Rsq_consolidate = [];
end
end