%% Load
clear all
close all

% Load the files provided by tools/evaluation.cpp
lc = csvread('output.txt');
runtime = csvread('runtime.txt');

%% Probability
haloc = zeros(1,5);
bow = zeros(1,5);
vlad = zeros(1,5);
for i=1:size(lc,1)
    if (lc(i,1) >= 0)
        haloc(lc(i,1)+1) = haloc(lc(i,1)+1) + 1;
    end
    if (lc(i,2) >= 0)
        bow(lc(i,2)+1) = bow(lc(i,2)+1) + 1;
    end
    if (lc(i,3) >= 0)
        vlad(lc(i,3)+1) = vlad(lc(i,3)+1) + 1;
    end
end

haloc = haloc./size(lc,1);
bow = bow./size(lc,1);
vlad = vlad./size(lc,1);

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
set(gca, 'FontSize', 35);
set(gca, 'FontName', 'Times');
xlabel('Top p candidates');
ylabel('Percentage of valid loop closings');

%% Time
min_val = 0;
max_val = size(runtime,1);
x_axis = min_val:1:max_val-1;

figure
plot(x_axis, runtime(:,1), 'Color',[0,0,0.561]);
hold on;
plot(x_axis, runtime(:,2), 'Color',[0.498,0,0], 'LineStyle', '--');
plot(x_axis, runtime(:,3), 'Color',[0,0.498,0], 'LineStyle', '--');
grid on;
legend('sisHALOC', 'BoW', 'VLAD');
set(gca, 'FontSize', 35);
set(gca, 'FontName', 'Times');
xlabel('Top p candidates');
ylabel('Percentage of valid loop closings');

fprintf('sisHALOC mean runtime %d\n', mean(runtime(:,1)));
fprintf('BoW mean runtime %d\n', mean(runtime(:,2)));
fprintf('VLAD mean runtime %d\n', mean(runtime(:,3)));
