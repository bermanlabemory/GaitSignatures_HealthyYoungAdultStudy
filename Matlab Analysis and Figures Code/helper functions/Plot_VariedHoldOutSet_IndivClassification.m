function [AccuracyDistribution] = Plot_VariedHoldOutSet_IndivClassification(Signatures)
% Using 3D kinemtatics PC and RNN signatures, vary the hold out set and
% plot the classification accuracy across different # of PCs]


classify_accuracy = cell(2,1);
std_accuracy = cell(2,1);


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


models_idx = [1,2,3,4]; % use the 3D kin gait and PC sigs

classification_run_store = [];
classification_run_store_std = [];
std_accuracy = [];
AccuracyDistribution = {};
average_accuracy = [];

for m = 1:length(models_idx); % # number of models

    for num_training_trials = 1:8;
        GaitSigs = Signatures{1,models_idx(m)}{1,1};
        Altered_SubjLabels = SubjectTrainTrials(keep_indices);
        Altered_GaitSigs = GaitSigs(keep_indices,:);
        Altered_sub_num = sub_num(keep_indices);

        accuracy_results = [];

        % Create a template for a linear SVM classifier
        template = templateSVM('Standardize', 1); % Standardize features

        for run = 1:num_runs
            rng(run); % Set a random seed for reproducibility

            train_data = [];
            train_labels = [];
            test_data = [];
            test_labels = [];

            %rng(run); %set a random seed for reproducibility

            subs = unique(Altered_sub_num);

            for subject = 1:num_subjects
                rng(run);
                shuffled_ind = randperm(num_trials_per_sub);

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
                accuracy_results =  [accuracy_results; accuracy];

            end

        end

        classification_run_store = [classification_run_store; mean(accuracy_results)]; %should get 8
        classification_run_store_std = [classification_run_store_std; std(accuracy_results)];

        AccuracyDistribution{num_training_trials,m} = accuracy_results;   

    end

    classify_accuracy{m,1} = classification_run_store;
    std_accuracy{m,1} = classification_run_store_std;

    classification_run_store =[];
    classification_run_store_std =[];
end

% consolidate data for plot
classify_data = [classify_accuracy{1,1} classify_accuracy{2,1} classify_accuracy{3,1} classify_accuracy{4,1}];
std_data = [std_accuracy{1,1} std_accuracy{2,1} std_accuracy{3,1} std_accuracy{4,1}];
titlelabel = ["2Dkinematics", "3Dkinematics", "3D kinetics", "alldata"];

% Labels for the models within each data type
%model_labels = {'RNN','PCA'};
model_labels = {'RNN'};

% Labels for the data types
data_type_labels = {'1', '2', '3', '4','5', '6 ','7','8'};


% Create a bar charts for each signature type
% for l = 1:4
% 
%     figure(l);
% 
%     b = bar(classify_data(:,l));
%     % Set labels and legend
%     xticklabels(data_type_labels);
%     xlabel('Number of Speed Trials in Training Set (per Individual)');
%     ylabel('Classification Accuracy (%)');
%     title(titlelabel(l));
% 
%     % Rotate x-axis labels for better readability (optional)
%     xtickangle(0);
%     legend(model_labels);
% 
%     hold on
%     % Add error bars
%     ngroups = size(classify_data(:,l), 2);
%     nbars = size(classify_data(:,l), 1);
%     groupwidth = min(0.8, nbars / (nbars + 1.5));
% 
%     for p = 1:nbars
%         x = p;
%         errorbar(x, classify_data(p,l), std_data(p, l), 'k', 'linestyle', 'none');
% 
%         % Add bar chart values to each bar
% 
%         text(x, classify_data(p, l) - 0.08, num2str(classify_data(p, l), '%.1f'), ...
%                 'HorizontalAlignment', 'center');
% 
%     end
% 
%     hold on
%     ylim([0,100])
%     hold on
% 
% 
% 
% end

end