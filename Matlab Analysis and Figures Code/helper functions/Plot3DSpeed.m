function [] = Plot3DSpeed(GaitSigs,condname,gaitgroup,spec,conditions)
% This function should take in the gait signature data and corresponding conditions and 
% generate 3D plots of the specified PCs (PC1, PC2,PC3) colored according to the
% conditions specified in the conditions vector
figure()
U = unique(conditions); %speeds
n_colors = length(U);
colors = jet(n_colors); % speed colors
for i = 1:length(U);
    %indices = find(U(i)== conditions & gaitgroup == spec); old dataset
    indices = find(U(i)== conditions); %YA_Dataset
    for k = 1:length(indices)
        plot3(GaitSigs(indices(k),1:100),GaitSigs(indices(k),101:200),GaitSigs(indices(k),201:300),'Color',colors(i,:),'LineStyle','-','LineWidth',2,'DisplayName',string(U(i)));
        hold on
    end
end
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
title([condname]) 
grid on
hold on
h = colorbar;
colormap(jet); 
clim([min(conditions),max(conditions)])
ylabel(h, 'speed (m/s)');
end

