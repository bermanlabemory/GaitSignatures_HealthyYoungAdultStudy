function [] = Plot3DConditions(GaitSigs,condname,gaitgroup,spec,conditions,pallette,plotting)
% This function should take in the gait signature data and corresponding conditions and 
% generate 3D plots of the specified PCs (PC1, PC2,PC3) colored according to the
% conditions specified in the conditions vector

set(groot,'DefaultFigureColor','w')
set(groot,'DefaultAxesFontSize',12)
set(groot,'DefaultAxesLineWidth',.5)
set(groot,'DefaultAxesFontName','Times New Roman')
set(groot,'DefaultLineLineWidth',1)
set(groot,'DefaultFigureUnits','inches')
set(groot,'DefaultFigurePosition',[.5, .5, 3.5, 2.5])

load('/Users/tanielwinner/Desktop/eLIFE_MATLAB_092221/YA_study/Reprocessed_HYA_092223/matfile extracted data and labels/sub_num_label.mat');

% set color pallette as either: "Blue", "Orange", "Red" , "Green", "Distinguish"
% Distinguish finds the most distinguishable colors given the number of
% different conditions needed to plot

%% Color Brewer colors
% Specify Color Palett
Blue = [8 48 107; 8 81 156; 33 113 181; 107 174 214;198 219 239; 66 146 198;158 202 225; 222 235 247; 218,218,235 ]./256; % blue shades

Blue2 = [255 255 217;
237 248 177;
199 233 180;
127 205 187;
65 182 196;
29 145 192;
34 94 168;
37 52 148;
8 29 88]./256;


Blue3 = [247 251 255;
222 235 247;
198 219 239;
158 202 225;
107 174 214;
66 146 198;
33 113 181;
8 81 156;
8 48 107]./256;

Red = [103 0 13;165 15 21;203 24 29; 239 59 44; 251 106 74; 252 146 114; 252 187 161]./256;%; 253 187 132]./256; %red shades

RedandBlue = [Blue;Red];

Purple = [239,237,245; %purple shades
218,218,235;
188,189,220;
158,154,200;
128,125,186;
106,81,163;
84,39,143;
63,0,125]./256;

Orange = [254,230,206 %orange shades
253,208,162
253,174,107
253,141,60
241,105,19
217,72,1
166,54,3
127,39,4]./256;

Green = [229,245,224  % green shades
199,233,192
161,217,155
116,196,118
65,171,93
35,139,69
0,109,44
0,68,27]./256;

Multi = [198, 219, 239;
    251 106 74;
    253,141,60]; % 3 indiv (blue, red, orange)

num_colors = length(GaitSigs);
% Create a colormap using parula with the desired number of colors
cmap = parula(num_colors);

 custom_colors =  [
    0, 0, 1;     % Blue
    1, 0, 0;     % Red
    0, 1, 1;     % Cyan
    1, 0, 1;     % Magenta
    0.5, 0, 0;   % Maroon
    1, 1, 0;     % Yellow
    0, 0, 0; % Black
     0, 0, 1;  % Royal Blue
    0, 0, 0.5;   % Navy
    1, 0.5, 0;   % Orange
     0.3,0,0.5;   % DarkIndigo  
    0, 0.5, 0;   % Dark Green
    0, 0.5, 1;   % Light Blue
    1, 0, 0.5;   % Dark Pink
     0.5, 0.5, 1; % Light Indigo
0.8, 0.6, 0.9; % Light Purple
    0.5, 0.5, 0; % Olive Green
    1, 0.5, 0.5; % Light Pink
    0.9, 0.9, 0.6; % Khaki
    0.5, 1, 0.5; % Light Green
];

 pointColors = custom_colors(sub_num, :); 
 
%%
figure()
U = unique(conditions);
n_colors = length(U); %how many unique colors do we need?

% Specifying color shades to use
if pallette == "Blue"
    colors = Blue;
elseif pallette == "Orange"
    colors = Orange;
elseif pallette == "Blue3"
    colors = Blue3;
elseif pallette == "Red"
    colors = Red;
elseif pallette == "RedandBlue"
    colors = RedandBlue;
elseif pallette == "Green"
    colors = Green;
elseif pallette == "Distinguish"
    bg = {'w'}; %set color of background for distinguishable_colors function (if that selection is made) default:white
    colors = distinguishable_colors(n_colors,bg); %use this for individuals
elseif pallette == "custom_colors"
    colors = custom_colors;
end


%% Plot the trials belonging to the same condition,the same color; different conditions different colors
c = 0;%1
for  r = 1:length(U);
    %indices = find(U(r)== conditions & gaitgroup == spec); % all indices for this person
    indices = find(U(r)== conditions); % all indices for this person -- altered for YAs
    if indices > 0;
        c= c+1; %update color
    end
    for k = 1:length(indices)
        plot3(GaitSigs(indices(k),1:100),GaitSigs(indices(k),101:200),GaitSigs(indices(k),201:300),'Color',pointColors(indices(k),:),'LineStyle','-','LineWidth',2,'DisplayName',string(U(r))) % for Python
        hold on
    end  
end
% xlim([-15 20])
% ylim([-15 20])
% zlim([-10 20])

% xlim([-20 20]) %previous
% ylim([-20 20])
% zlim([-15 20])

% xlim([-20 20])
% ylim([-20 20])
% zlim([-15 20])

xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
title([condname]) 
grid on
end

