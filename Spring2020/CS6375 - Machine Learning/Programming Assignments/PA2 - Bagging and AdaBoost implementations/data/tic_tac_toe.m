clear;

% Load the car data set and split into train and test sets
fID = fopen('./tic-tac-toe.data', 'r');
M = textscan(fID, '%s%s%s%s%s%s%s%s%s%s', 'delimiter', ',');
fclose(fID);

nCols = length(M);
nRows = length(M{1});
Z = nan(nRows, nCols);
for j = 1:nCols
    [~, ~, Z(:, j)] = unique(M{j});
end

Z = Z - 1;
y = Z(:, 10);
x = Z(:, 1:9);

f = 4; % One in 4 examples is a test example
yValues = unique(y);
xTrn = []; yTrn = [];
xTst = []; yTst = [];
for i = 1:length(yValues)
    I = find(y == yValues(i));
    xc = x(I, :);
    [L, N] = size(xc);
    
    Itst = I(1:f:end);
    Itrn = setdiff(I, Itst);
    
    Ltst = length(Itst);
    Ltrn = length(Itrn);
    
    xTrn = [xTrn; x(Itrn, :)];
    xTst = [xTst; x(Itst, :)];
    
    yTrn = [yTrn; (i - 1) * ones(Ltrn, 1)];
    yTst = [yTst; (i - 1) * ones(Ltst, 1)];
end

% Full data sets
Itrn = randperm(length(yTrn));
Itst = randperm(length(yTst));


csvwrite('./tic-tac-toe.train', [yTrn(Itrn) xTrn(Itrn, :)]);
csvwrite('./tic-tac-toe.test', [yTst(Itst) xTst(Itst, :)]);