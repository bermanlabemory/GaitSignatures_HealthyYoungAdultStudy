addpath '/Users/tanielwinner/Desktop/eLIFE_MATLAB_092221/'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set parameters
set(groot,'DefaultFigureColor','w')
set(groot,'DefaultAxesFontSize',18)
% set(groot,'DefaultAxesLineWidth',.5)
set(groot,'DefaultAxesFontName','Arial')
% set(groot,'DefaultLineLineWidth',1)
 set(groot,'DefaultFigureUnits','inches')
% set(groot,'DefaultFigurePosition',[.5, .5, 3.5, 2.5])


%%
%load('ClinDatas.mat');
load('/Users/tanielwinner/Desktop/eLIFE_MATLAB_092221/YA_study/Reprocessed_HYA_092223/matfile extracted data and labels/HYA_Study_InformationTable_111523.mat')
names = HYA_Study_InformationTable.Properties.VariableNames;
%mask = ~(isnan(HYA_Study_InformationTable{:, 18}) | isnan(HYA_Study_InformationTable.SS_Fast)); %exclude where balance beam missing

for i = 2:49   %column variables of interest
    y1 = HYA_Study_InformationTable{:, i};
    
    if iscategorical(y1)
        continue

    elseif iscell(y1)
        y2 = cell2mat(y1);
        mask = ~(isnan(y2) | isnan(HYA_Study_InformationTable.SS_Fast));
        switch i
            case 18 % balance beam
                mask(14) = false;
                mask(17) = false;
                mask(19) = false;
            case {38, 39, 40, 41} % left ankle
                mask(7) = false;
                mask(10) = false;
            case {29,31,33,35} % right hip
                mask(20) = false;
        end

        fast = HYA_Study_InformationTable{mask, 46};% 46
        slow = HYA_Study_InformationTable{mask, 47}; %47
        y = y2(mask);
        speedconditions = HYA_Study_InformationTable{mask, 3};
        idx = find(speedconditions == 8); % SS speeds only
        name = names{i};

        corrPlot(fast(idx), y(idx), name);
        % corrPlot(slow, y, name);

    else   
        y2 = y1;
        mask = ~(isnan(y2) | isnan(HYA_Study_InformationTable.SS_Fast));
        switch i
            case 18 % balance beam
                mask(14) = false;
                mask(17) = false;
                mask(19) = false;
            case {38, 39, 40, 41} % left ankle
                mask(7) = false;
                mask(10) = false;
            case {29,31,33,35} % right hip
                mask(20) = false;
        end

        fast = HYA_Study_InformationTable{mask, 48};% 46 gait sigs, 48 pca sigs
        %slow = HYA_Study_InformationTable{mask, 47}; %47
        y = y2(mask);
        speedconditions = HYA_Study_InformationTable{mask, 3};
        idx = find(speedconditions == 8); % SS speeds only
        name = names{i};

        corrPlot(fast(idx), y(idx), name);
        % corrPlot(slow, y, name);

    end
end

function [] = corrPlot(x, y, name)
[r, p_value] = corrcoef(x, y);
coefficients = polyfit(x, y, 1);
x_values = linspace(min(x), max(x), 100);
y_values = polyval(coefficients, x_values);
r_value = r(1, 2);
p_value = p_value(1, 2);

% Plot the scatter plot
figure();
scatter(x, y, 100, 'filled');
xlabel('Distance');
ylabel(name);
hold on;

% Plot the regression line
plot(x_values, y_values, 'r', 'LineWidth', 2);

text(0.5, 0.9, sprintf('Correlation Coefficient: %.2f\np-value: %.4f', r_value, p_value), ...
    'Units', 'normalized', 'FontSize', 15, 'FontWeight', 'bold');
legend('Data', 'Linear Fit', 'Location', 'NorthWest');
end

