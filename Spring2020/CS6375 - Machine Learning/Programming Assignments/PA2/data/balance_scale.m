clear;

% Load the balance-scale data set and split into train and test sets
fID = fopen('./balance-scale.data', 'r');
M = textscan(fID, '%s%d%d%d%d', 'delimiter', ',');
fclose(fID);

yStr = M{1};
x = cell2mat(M(2:5));


yValues = unique(yStr);
xTrn = [];
yTrn = [];
xTst = [];
yTst = [];

f = 4; % One in 4 examples is a test example
for i = 1:length(yValues)
    I = find(strcmpi(yStr, yValues{i}));
    xc = x(I, :);
    [L, N] = size(xc);
    
    Itst = I(1:f:end);
    Itrn = setdiff(I, Itst);
    
    Ltst = length(Itst);
    Ltrn = length(Itrn);
    
    xTrn = [xTrn; x(Itrn, :) - 1];
    xTst = [xTst; x(Itst, :) - 1];
    
    yTrn = [yTrn; (i - 1) * ones(Ltrn, 1)];
    yTst = [yTst; (i - 1) * ones(Ltst, 1)];
end

% Full data sets
Itrn = randperm(length(yTrn));
Itst = randperm(length(yTst));


csvwrite('./balance-scale.train', [yTrn(Itrn) xTrn(Itrn, :)]);
csvwrite('./balance-scale.test', [yTst(Itst) xTst(Itst, :)]);
