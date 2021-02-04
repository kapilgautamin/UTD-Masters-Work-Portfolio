% Load the car data set and split into train and test sets
fID = fopen('./agaricus-lepiota.data', 'r');

N = 23;
formatStr = repmat('%s', 1, N);
M = textscan(fID, formatStr, 'delimiter', ',');
fclose(fID);

% Drop all columns with missing features
Idrop = false(N, 1);
for i = 1:N
    v = unique(M{:, i});
    if any(strcmpi(v, '?')), Idrop(i) = true; end
end
M(Idrop) = [];

nCols = length(M);
nRows = length(M{1});
Z = nan(nRows, nCols);
for j = 1:nCols
    [~, ~, Z(:, j)] = unique(M{j});
end

Z = Z - 1;
y = Z(:, 5);
x = Z(:, [1:4 6:end]);

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


csvwrite('./mushroom.train', [yTrn(Itrn) xTrn(Itrn, :)]);
csvwrite('./mushroom.test', [yTst(Itst) xTst(Itst, :)]);