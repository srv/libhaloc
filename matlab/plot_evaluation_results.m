%% Load
clear all
close all

M = csvread('output.txt');
[rows columns] = size(M);

%% Probability
haloc = zeros(1,5);
bow = zeros(1,5);
vlad = zeros(1,5);
for i=1:rows
    if (M(i,1) >= 0)
        haloc(M(i,1)+1) = haloc(M(i,1)+1) + 1;
    end
    if (M(i,2) >= 0)
        bow(M(i,2)+1) = bow(M(i,2)+1) + 1;
    end
    if (M(i,3) >= 0)
        vlad(M(i,3)+1) = vlad(M(i,3)+1) + 1;
    end
end

haloc = haloc./size(M,1);
bow = bow./size(M,1);
vlad = vlad./size(M,1);

sum_haloc = 0;
sum_bow = 0;
sum_vlad = 0;
for i=1:size(haloc,2)
    sum_haloc = sum_haloc + haloc(i);
    sum_bow = sum_bow + bow(i);
    sum_vlad = sum_vlad + vlad(i);
    haloc(i) = sum_haloc;
    bow(i) = sum_bow;
    vlad(i) = sum_vlad;
end

figure
bar([haloc', bow', vlad']);
grid on;
legend('sisHALOC', 'BoW', 'VLAD');

%% Time
min_val = 0;
max_val = 2500;
step = (max_val - min_val)/rows;
x_axis = min_val:step:max_val-step;

figure
plot(x_axis, M(:,4), 'Color',[0,0,0.561]);
hold on;
plot(x_axis, M(:,5), 'Color',[0.498,0,0], 'LineStyle', '--');
plot(x_axis, M(:,6), 'Color',[0,0.498,0], 'LineStyle', '--');
grid on;
legend('sisHALOC', 'BoW', 'VLAD');

fprintf('sisHALOC mean runtime %d\n', mean(M(:,4)));
fprintf('BoW mean runtime %d\n', mean(M(:,5)));
fprintf('VLAD mean runtime %d\n', mean(M(:,6)));