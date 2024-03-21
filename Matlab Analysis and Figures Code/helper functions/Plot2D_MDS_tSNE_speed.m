function [] = Plot2D_MDS_tSNE_speed(GaitSigs,conditions)
% This function plots the tSNE representation of the gait signatures and
% colors it according to the 

%% 2D tsne
% figure()
% conditions = conditions*0.01;
% %colormap parula
% rng default % for reproducibility
% U = unique(conditions);
% % get distinguishable colors
% bg = {'w'};
% n_colors = length(U);
% colors = jet(n_colors);
% Y = tsne(GaitSigs,'NumPCAComponents',6,'Perplexity',20);
% %markers = ['o';'+';'*';'.';'x';'s';'d';'^';'p';'h';'_';'|';'>'];
% gscatter(Y(:,1),Y(:,2),conditions,colors,'.',40);
% hold on
% title('tSNE speed')
% axis equal
% c = colorbar;
% caxis([min(conditions) max(conditions)]);
% c.Location = 'eastoutside';
%print -painters -depsc tsnespeed.eps
%% 2D MDS
% figure()
% rng default % for reproducibility
% U = unique(conditions);
% % get distinguishable colors
% bg = {'w'};
% n_colors = length(U);
% colors = jet(n_colors);
% D = pdist2(GaitSigs,GaitSigs,'euclidean'); % dissimilarity matrix
% Y = mdscale(D,2);
% %markers = ['o';'+';'*';'.';'x';'s';'d';'^';'p';'h';'_';'|';'>'];
% gscatter(Y(:,1),Y(:,2),conditions,colors,'.',40);
% hold on
% title('MDS speed')
% axis equal
% c = colorbar;
% caxis([min(conditions) max(conditions)]);
% c.Location = 'eastoutside';

%% Plot 3D MDS according to speed 
AB_indices = [1:30];
AB_sigs = GaitSigs(AB_indices,:);

% use kmeans to get centroid
[ind,centroid] = kmeans(AB_sigs,1);

% concatenate the centroid to all the trials 
All_GS = [GaitSigs;centroid];

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
for n = 1:72 % number of individuals
    c = c+1;
    f = find(conditions(n) == U); % find the index color to use
    c1 = colors(f,1);
    c2 = colors(f,2);
    c3 = colors(f,3);
    S = c;
    scatter3(Y2(n,1),Y2(n,2),Y2(n,3),'MarkerFaceColor',[c1,c2,c3], 'MarkerEdgeColor',[0,0,0],'MarkerEdgeColor',[0,0,0]); % specify color and markersize
    hold on
    if n ==6
       c = 0;
    end

end

%scatter3(Y2(:,1),Y2(:,2),Y2(:,3),colors,'.',100);

%text(Y(:,1),Y(:,2),annotatedlabels,'FontSize', 12,'Color','k','HorizontalAlignment','right')
hold on
plot3(Y2(73,1),Y2(73,2),Y2(73,3), 'ks', 'MarkerSize', 35, 'MarkerFaceColor','k');
title('MDS Speed')
axis equal
grid on
%grid(gca,'minor')
c = colorbar();
caxis([min(conditions) max(conditions)]);
c.Location = 'eastoutside';

% figure(3)
% set(gcf,'Renderer','Painter')
% hgexport(gcf,'3D_MDS_speed.eps')



end

