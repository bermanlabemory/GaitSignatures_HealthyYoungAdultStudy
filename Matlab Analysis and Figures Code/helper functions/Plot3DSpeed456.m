function [] = Plot3DSpeed456(GaitSigs,condname,gaitgroup,spec,conditions)
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
        plot3(GaitSigs(indices(k),301:400),GaitSigs(indices(k),401:500),GaitSigs(indices(k),501:600),'Color',colors(i,:),'LineStyle','-','LineWidth',2,'DisplayName',string(U(i)));
        hold on
    end
end
xlabel('PC 4')
ylabel('PC 5')
zlabel('PC 6')
title([condname]) 
grid on
hold on
h = colorbar;
colormap(jet); 
clim([min(conditions),max(conditions)])
ylabel(h, 'speed (m/s)');
end

