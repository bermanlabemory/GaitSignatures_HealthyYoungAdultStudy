function [] = corrPlot(x, y, name, colors)

% Set parameters
set(groot,'DefaultFigureColor','w')
set(groot,'DefaultAxesFontSize',18)
% set(groot,'DefaultAxesLineWidth',.5)
set(groot,'DefaultAxesFontName','Arial')
% set(groot,'DefaultLineLineWidth',1)
 set(groot,'DefaultFigureUnits','inches')
% set(groot,'DefaultFigurePosition',[.5, .5, 3.5, 2.5])

[r, p_value] = corrcoef(x, y);
coefficients = polyfit(x, y, 1);
x_values = linspace(min(x), max(x), 100);
y_values = polyval(coefficients, x_values);
r_value = r(1, 2);
p_value = p_value(1, 2);

% Plot the scatter plot
figure();
scatter(x, y, 100, colors, 'filled');
xlabel('Distance');
ylabel(name);
hold on;

% Plot the regression line
plot(x_values, y_values, 'r', 'LineWidth', 2);

text(0.5, 0.9, sprintf('Correlation Coefficient: %.2f\np-value: %.4f', r_value, p_value), ...
    'Units', 'normalized', 'FontSize', 15, 'FontWeight', 'bold');
legend('Data', 'Linear Fit', 'Location', 'NorthWest');
end