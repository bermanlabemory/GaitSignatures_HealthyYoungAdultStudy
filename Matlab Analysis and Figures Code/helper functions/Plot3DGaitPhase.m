function  Plot3DGaitPhase(gaitgroupindices, GaitSigs, gaitphase,plotcondition)
% Plot 3D loops with respect to gait phase
figure()
for k = 1:length(gaitgroupindices);
    r = gaitgroupindices(k);
    plot3(GaitSigs(r,1:100),GaitSigs(r,101:200),GaitSigs(r,201:300),'k');
    hold on  
    for j = 1: 100 % length of trial (phase averaged)
        hold on
        %plot one trial
        if gaitphase(r,j) == 1
            %caption = 'Left Swing';
            plot3(GaitSigs(r,j),GaitSigs(r,j+100),GaitSigs(r,j+200),'b.','MarkerSize',8,'DisplayName','Left Swing');
            hold on;
        elseif gaitphase(r,j) == 2
            %caption = 'Left Stance';
            hold on
            plot3(GaitSigs(r,j),GaitSigs(r,j+100),GaitSigs(r,j+200),'c.','MarkerSize',8,'DisplayName','Left Stance');
            hold on
        elseif gaitphase(r,j) == 3
            %caption = 'Right Swing';
            plot3(GaitSigs(r,j),GaitSigs(r,j+100),GaitSigs(r,j+200),'g.','MarkerSize',8,'DisplayName','Right Swing');
            hold on
        elseif gaitphase(r,j) == 4
            %caption = 'Right Stance';
            plot3(GaitSigs(r,j),GaitSigs(r,j+100),GaitSigs(r,j+200),'m.','MarkerSize',8,'DisplayName','Right Stance');
            hold on
        end
    end
end
hold off
title([plotcondition])
grid on
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')

xlim([-20 20])
ylim([-20 20])
zlim([-15 20])
legendUnq()
legend()

end

