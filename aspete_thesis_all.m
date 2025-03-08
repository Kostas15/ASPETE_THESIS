%% Create a cell array with the filenames
filenames = {'car1.mat', 'car3.mat', 'car4.mat', ...
             'car6.mat', 'car7.mat', 'car8.mat', 'car9.mat', ...
             'car11.mat', 'car12.mat', 'car13.mat'};

% Determine the number of files to select randomly
numFiles = 10; 

% Randomly select file indices
selectedIndices = 1:10;

%% Load and select training data
traingingTable = [];
testingTable = [];
trainingIndicesPerFile = cell(numFiles, 1); % To store indices for each file
for i = 1:numFiles
    selectedFile = filenames{selectedIndices(i)};
    
    data = load(selectedFile);
    dataMatrix = data.vehicle_data;
    [numRows, ~] = size(dataMatrix);

    trainingSize = round(0.8 * numRows);
    
    startIdx1 = randi([1, numRows - trainingSize + 1]);
    trainingChunk = dataMatrix(startIdx1:startIdx1 + trainingSize - 1, :);
    
    trainingIndicesPerFile{i} = [trainingIndicesPerFile{i}; startIdx1, startIdx1 + trainingSize - 1];
    
    traingingTable = [traingingTable; trainingChunk];

    % Extract the remaining 20% of the data for testing
    testingChunk = dataMatrix;
    testingChunk(startIdx1:startIdx1 + trainingSize - 1, :) = []; % Remove training data
    testingTable = [testingTable; testingChunk];
end

% Select predictors and Labels
X1 = traingingTable(:, 9);
X2 = traingingTable(:, 10);
X3 = traingingTable(:, 11);
X4 = traingingTable(:, 15);
XTrain = [X1 X2 X3 X4];
YTrain = traingingTable(:, 7);

% Training
Model = fitcensemble(XTrain, YTrain, 'Method', 'Bag');

%% Load chunk for testing
ind = unique(testingTable.VEHICLE_ID);
carLabel = ind(9);
carData = testingTable(strcmp(testingTable.VEHICLE_ID, carLabel), :);

numCarRows = height(carData);
startIdx2 = randi([1, numCarRows - 99]);
chunk = carData(startIdx2:startIdx2 + 99, :);

testingTb = chunk;

X5 = testingTb(:, 9);
X6 = testingTb(:, 10);
X7 = testingTb(:, 11);
X8 = testingTb(:, 15);
XTest = [X5 X6 X7 X8];
YTest = testingTb(:, 7); % Labels

% Testing
YPred = predict(Model, XTest);

% Display accuracy
YPredLabels = YPred;
YTestLabels = YTest.VEHICLE_ID;

accuracy = zeros(size(YTestLabels));

for i = 1:numel(YTestLabels)
    accuracy(i) = strcmp(YPredLabels{i}, YTestLabels{i});
end

overallAccuracy = mean(accuracy) * 100; 
fprintf('Overall Accuracy of %s: %.0f%%\n', carLabel, overallAccuracy);