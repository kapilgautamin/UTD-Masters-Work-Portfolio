clear;
close all;

M = csvread('mushroom.train');
x = M(:, 2:end);
y = M(:, 1);



depth = 5;
t = fitctree(x, y, 'MinLeafSize', 2^depth, 'AlgorithmForCategorical', 'Exact',...
        'PredictorSelection', 'allsplits');
view(t, 'Mode', 'Graph')

yTest = predict(t, x);
err = mean(y ~= yTest);
confusionmat(y, yTest)

% p = arrayfun(@(v) sum(y == v), unique(y)) / length(y); 
% ent = -p' * log2(p);
% 
% for attr = 1:size(x, 2)
%     for value = unique(x(:, attr))'
%         I = x(:, attr) == value;
%         cEnt = crossentropy(I, y);
%         mi = ent - cEnt;
%         
%         fprintf('mi(attr_%d = %d) = %g (%d points).\n', attr, value, mi, sum(I));
%     end
% end