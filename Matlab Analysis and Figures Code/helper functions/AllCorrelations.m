function [] = AllCorrelations(table)
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
names = table.Properties.VariableNames;
%mask = ~(isnan(HYA_Study_InformationTable{:, 18}) | isnan(HYA_Study_InformationTable.SS_Fast)); %exclude where balance beam missing

for i = 4:45   %column variables of interest
    y1 = table{:, i};
    
    if iscategorical(y1)
        continue

    elseif iscell(y1)
        y2 = cell2mat(y1);
        mask = ~(isnan(y2) | isnan(table.SS_Slow));
        switch i
            % case 18 % balance beam
            %     mask(89:97) = false; %YA014
            %     mask(115:123) = false; %YA017
            %     mask(133:141) = false; %YA019
            case {38, 39, 40, 41} % left ankle injury
                mask(35:43) = false; %YA007
                %mask(62:70) = false; % YA010
            case {29,31,33,35} % right hip injury
                %mask(142:150) = false; %YA020
        end
    
        %%%% ~~~~~~CHANGE d_var to the dependent variable and which speed
        %%%% condition to plot ~~~~~~~
      
        d_var = 46; % 2 - speed %Dependent Vars: 46- SS_Slow, 47- SS_Fast, 48-Slow-Fast
        sp_cond = 3; %SS speeds
        x_label = 'Distance';
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        x = table{mask, d_var};% %account for speed
        %x = cell2mat(table{mask, 2})*0.01; %speed dependent var
        speedconditions = table{mask, 3}; %don't change trialtype is col 3
        idx = find(speedconditions == sp_cond); % trial speed to plot values for


        % don't change here
        y = y2(mask);
        colors = table{mask,49}; %colors

        name = names{i}; %variable names to plot on y axis
        corrPlot(x(idx), y(idx), name, colors(idx,:),x_label);

    else   
        y2 = y1;
        mask = ~(isnan(y2) | isnan(table.SS_Slow));
        switch i
            % case 18 % balance beam
            %     mask(89:97) = false; %YA014
            %     mask(115:123) = false; %YA017
            %     mask(133:141) = false; %YA019
            case {38, 39, 40, 41} % left ankle injury
                mask(35:43) = false; %YA007
                %mask(62:70) = false; % YA010
            case {29,31,33,35} % right hip injury
                %mask(142:150) = false; %YA020
        end


        x = table{mask, d_var}; %Dependent Vars: 46- SS_Slow, 47- SS_Fast, 48-Slow-Fast
        %x = cell2mat(table{mask, 2})*0.01;% speed in particular
        speedconditions = table{mask, 3}; %don't change trial type is column 3
        idx = find(speedconditions == sp_cond); % SS speed
        
        % don't change here
        colors = table{mask,49}; %colors
        y = y2(mask);
        name = names{i};
        corrPlot(x(idx), y(idx), name, colors(idx,:),x_label);


    end
end

function [] = corrPlot(x, y, name, colors, x_label)
[r, p_value] = corrcoef(x, y);
coefficients = polyfit(x, y, 1);
x_values = linspace(min(x), max(x), 100);
y_values = polyval(coefficients, x_values);
r_value = r(1, 2);
p_value = p_value(1, 2);

% Plot the scatter plot
figure();
scatter(x, y, 200, colors, 'filled');
xlabel(x_label);
ylabel(name);
hold on;

% Plot the regression line
plot(x_values, y_values, 'r', 'LineWidth', 2);

text(0.5, 0.9, sprintf('Correlation Coefficient: %.2f\np-value: %.4f', r_value, p_value), ...
    'Units', 'normalized', 'FontSize', 15, 'FontWeight', 'bold');
legend('Data', 'Linear Fit', 'Location', 'NorthWest');
end


end