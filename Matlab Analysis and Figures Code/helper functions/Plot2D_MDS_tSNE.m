function [] = Plot2D_MDS_tSNE(GaitSigs,conditions,pallette,mkrsize,sub_num)
% This function plots the tSNE representation of the gait signatures and
% colors it according to the 

%% Color Brewer colors
% Specify Color Palett
%Blue = [8 48 107; 8 81 156; 33 113 181; 107 174 214;198 219 239]./256; % blue shades
Blue = [8 48 107; 8 81 156; 33 113 181; 107 174 214;198 219 239; 66 146 198;158 202 225; 222 235 247; 218,218,235 ]./256; % blue shades

Red = [103 0 13;165 15 21;203 24 29; 239 59 44; 251 106 74; 252 146 114; 252 187 161]./256;% 253 187 132]./256; %red shades

RedandBlue = [Blue;Red];



%%
U = unique(conditions);
n_colors = length(U); %how many unique colors do we need?

% Specifying color shades to use
if pallette == "Blue"
    colors = Blue;
elseif pallette == "Orange"
    colors = Orange;
elseif pallette == "Red"
    colors = Red;
elseif pallette == "RedandBlue"
    colors = RedandBlue;
elseif pallette == "Green"
    colors = Green;
elseif pallette == "Distinguish"
    bg = {'w'}; %set color of background for distinguishable_colors function (if that selection is made) default:white
    colors = distinguishable_colors(n_colors,bg); %use this for individuals
end

%% tsne
figure()
rng default % for fair comparison
Y = tsne(GaitSigs,'NumPCAComponents',6,'Perplexity',20);
%markers = ['o';'+';'*';'.';'x';'s';'d';'^';'p';'h';'_';'|';'>'];
gscatter(Y(:,1),Y(:,2),conditions,colors,'.',40);
hold on
%text(Y(:,1),Y(:,2),annotatedlabels,'FontSize', 12,'Color','k','HorizontalAlignment','right')
title('tSNE')
axis equal


%% MDS
rng default % for fair comparison
D = pdist2(GaitSigs,GaitSigs,'euclidean'); % dissimilarity matrix
Y = mdscale(D,3);
figure()
gscatter(Y(:,1),Y(:,2),conditions,colors,'.',40);
hold on
%text(Y(:,1),Y(:,2),annotatedlabels,'FontSize', 12,'Color','k','HorizontalAlignment','right')
title('MDS')
axis equal

%% Additional processing to add centroid to plot
%hold on
%AB_indices = [1:30];
%AB_sigs = GaitSigs(AB_indices,:);

% use kmeans to get centroid
%[ind,centroid] = kmeans(AB_sigs,1);

% concatenate the centroid to all the trials 
%All_GS = [GaitSigs;centroid];

% Create a dissimilarity matrix.
%D = pdist2(All_GS,All_GS,'euclidean'); % dissimilarity matrix
%Y2 = mdscale(D,2); % get the centroid in 2D for plotting

%plot the centroid in a big black dot
%plot(Y2(73,1),Y2(73,2), 'k.', 'MarkerSize', 80); % index 73 is centroid

%% plot speeds different marker shapes
figure()
c = 0; % counter variable
for n = 1:length(U) % number of individuals
    for j = 1:6 % number of speeds
        c = c+1;
        c1 = colors(n,1);
        c2 = colors(n,2);
        c3 = colors(n,3);
        scatter(Y(c,1),Y(c,2),mkrsize(c),'MarkerFaceColor',[c1,c2,c3], 'MarkerEdgeColor',[0,0,0]); % specify color and markersize
        hold on
    end
end
%text(Y(:,1),Y(:,2),annotatedlabels,'FontSize', 12,'Color','k','HorizontalAlignment','right')
title('MDS Speed')
axis equal

% Additional processing to add centroid to plot
hold on
AB_indices = [1:30];
AB_sigs = GaitSigs(AB_indices,:);

% use kmeans to get centroid
[ind,centroid] = kmeans(AB_sigs,1);

% concatenate the centroid to all the trials 
All_GS = [GaitSigs;centroid];

% Create a dissimilarity matrix.
D = pdist2(All_GS,All_GS,'euclidean'); % dissimilarity matrix
Y2 = mdscale(D,2); % get the centroid in 2D for plotting

%plot the centroid in a big black dot
%plot(Y2(73,1),Y2(73,2), 'kd', 'MarkerSize', 50, 'MarkerFaceColor','k'); % index 73 is centroid

%xlim([-60 60])
%ylim([-80 80])

%% 3D MDS all gray with centroid square
figure()
AB_indices = [1:30];
AB_sigs = GaitSigs(AB_indices,:);

% use kmeans to get centroid
[ind,centroid] = kmeans(AB_sigs,1);

% concatenate the centroid to all the trials 
All_GS = [GaitSigs;centroid];

% Create a dissimilarity matrix.
D = pdist2(All_GS,All_GS,'euclidean'); % dissimilarity matrix
Y2 = mdscale(D,3); % get the centroid in 2D for plotting


plot3(Y2(1:end-1,1),Y2(1:end-1,2),Y2(1:end-1,3),'.','Color',[.7,.7,.7],'MarkerSize',75);
hold on
%plot the centroid in a big black dot
plot3(Y2(73,1),Y2(73,2),Y2(73,3), 'ks', 'MarkerSize', 35, 'MarkerFaceColor','k','MarkerEdgeColor','k'); % index 73 is centroid
axis equal
grid on
grid(gca,'minor')
%% 3D MDS colored 
% figure(51)
% colorMDS = [];
% for f = 1:length(RedandBlue)
%    for g = 1:6
%        colorMDS = [colorMDS; RedandBlue(f,:)];
%    end
% end
% 
% plot3(Y2(1:end-1,1),Y2(1:end-1,2),Y2(1:end-1,3),'.','MarkerSize',75);
% % for k = 1:length(Y2)-1
% %     scatter3(Y2(k,1),Y2(k,2),Y2(k,3),'.','Color',colorMDS(k,:));
% %     hold on
% % end
% hold on
% %text(Y(:,1),Y(:,2),annotatedlabels,'FontSize', 12,'Color','k','HorizontalAlignment','right')
% title('MDS')
% axis equal



%% 3D MDS according to color and speed
figure()
c = 0; % counter variable
for n = 1:length(U) % number of individuals
    for j = 1:6 % number of speeds
        c = c+1;
        c1 = colors(n,1);
        c2 = colors(n,2);
        c3 = colors(n,3);
        scatter3(Y2(c,1),Y2(c,2),Y2(c,3),mkrsize(j),'MarkerFaceColor',[c1,c2,c3], 'MarkerEdgeColor',[0,0,0]); % specify color and markersize
        hold on
    end
end
%text(Y(:,1),Y(:,2),annotatedlabels,'FontSize', 12,'Color','k','HorizontalAlignment','right')
hold on
plot3(Y2(73,1),Y2(73,2),Y2(73,3), 'ks', 'MarkerSize', 35, 'MarkerFaceColor','k');
title('MDS Speed')
axis equal
grid on
grid(gca,'minor')

%% Plot 3D MDS according to AB, high, low funct

% indices
ABi = [1:30];
HFi = [31:36,55:60,67:72];
LFi = [37:54,61:66];

% colors for the different subgroups
clrAB = [8, 48, 107]./255; %blue
clrHF = [203, 24, 29]./255; %red
clrLF = [253,141,60]./255; %orange

figure()
c = 0; % counter variable
for n = 1:length(U) % number of individuals
    for j = 1:6 % number of speeds
        c = c+1;
        if ismember(c,ABi) 
            scatter3(Y2(c,1),Y2(c,2),Y2(c,3),mkrsize(j),'MarkerFaceColor',clrAB, 'MarkerEdgeColor',[0,0,0]); % specify color and markersize
            hold on
        elseif ismember(c,HFi) 
            scatter3(Y2(c,1),Y2(c,2),Y2(c,3),mkrsize(j),'MarkerFaceColor',clrHF, 'MarkerEdgeColor',[0,0,0]); % specify color and markersize
            hold on
        elseif ismember(c,LFi)
            scatter3(Y2(c,1),Y2(c,2),Y2(c,3),mkrsize(j),'MarkerFaceColor',clrLF, 'MarkerEdgeColor',[0,0,0]); % specify color and markersize
            hold on
        end
    end
end
%text(Y(:,1),Y(:,2),annotatedlabels,'FontSize', 12,'Color','k','HorizontalAlignment','right')
hold on
plot3(Y2(73,1),Y2(73,2),Y2(73,3), 'ks', 'MarkerSize', 35, 'MarkerFaceColor','k');
title('MDS Speed')
axis equal
grid on
grid(gca,'minor')


end