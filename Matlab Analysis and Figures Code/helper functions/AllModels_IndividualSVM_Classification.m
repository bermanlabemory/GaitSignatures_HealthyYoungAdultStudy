function [average_accuracy,std_accuracy, AccuracyDistribution] = AllModels_IndividualSVM_Classification(Signatures)

% Random Train-Test Split Classification: Train SVM classifier where 3 random speeds from each subject is selected as test set and 6 are used for training
% Total of 25 runs, rng set on each run to keep same across model runs
% since some individuals do not have all 9 speed conditions - we will leave out 3
%
% individuals from this test: YA004,YA015,YA006

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
num_training_trials = 4; %change here
average_accuracy = [];
std_accuracy = [];

AccuracyDistribution = {};

for m = 1:8; % # number of models 

    GaitSigs = Signatures{1,m}{1,1}; %gait sig type
    Altered_SubjLabels = SubjectTrainTrials(keep_indices);
    Altered_GaitSigs = GaitSigs(keep_indices,:);
    Altered_sub_num = sub_num(keep_indices);

    accuracy_results = []; %zeros(num_runs*num_subjects,1);
   
    % Create a template for a linear SVM classifier
    template = templateSVM('Standardize', 1); % Standardize features
    
    for run = 1:num_runs  
        rng(run);
        train_data = [];
        train_labels = [];
        test_data = [];
        test_labels = [];
    
    
        %rng(run); %set a random seed for reproducibility
        
        subs = unique(Altered_sub_num);
    
        for subject = 1:num_subjects
        %rng(run); % Set a random seed for reproducibility each run    
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
        accuracy = sum(predicted_labels == test_labels) / numel(test_labels)*100;
        
        
        % Store accuracy in results
        accuracy_results = [ accuracy_results; accuracy];
        
        end

    end
    AccuracyDistribution{1,m} = accuracy_results;
    average_accuracy = [average_accuracy mean(accuracy_results(:))];
    std_accuracy = [std_accuracy std(accuracy_results(:))];
    
end


classify_data = [average_accuracy(5) average_accuracy(1); average_accuracy(6) average_accuracy(2); average_accuracy(7) average_accuracy(3); average_accuracy(8) average_accuracy(4)]; % consolidated from other script 

std_data = [std_accuracy(5) std_accuracy(1); std_accuracy(6) std_accuracy(2); std_accuracy(7) std_accuracy(3); std_accuracy(8) std_accuracy(4)]./sqrt(42); % Standard error of the mean calculation, 42 is the number of samples predicted in each test set

% Number of data types and PCs
num_data_types = size(classify_data, 2);

% Labels for the models within each data type
model_labels = {'PCA', 'RNN'};

% Labels for the data types
data_type_labels = {'2D Kinematics', '3D Kinematics', 'Kinetics', 'All Data'};

% Create a bar chart to visualize the variance accounted for by the 1st 6 PCs
figure;
b = bar(classify_data);

% Set labels and legend
xticklabels(data_type_labels);
xlabel('Data Types');
ylabel('Average SVM Classification Accuracy (%)');
title('Averaged SVM Individual Classification Accuracy over 10 runs [6 train/3 test trial split per subject] for Different Data Types and Models');

% Rotate x-axis labels for better readability (optional)
xtickangle(45);
legend(model_labels);

hold on
% Add error bars
ngroups = size(classify_data, 1);
nbars = size(classify_data, 2);
groupwidth = min(0.8, nbars / (nbars + 1.5));

for i = 1:nbars
    x = (1:ngroups) - groupwidth / 2 + (2 * i - 1) * groupwidth / (2 * nbars);
    errorbar(x, classify_data(:, i), std_data(:, i), 'k', 'linestyle', 'none');

     % Add bar chart values to each bar
    for j = 1:numel(classify_data(:, i))
        text(x(j), classify_data(j, i) - 0.08, num2str(classify_data(j, i), '%.2f'), ...
            'HorizontalAlignment', 'center');
    end

end


end