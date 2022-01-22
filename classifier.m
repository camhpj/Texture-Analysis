%% Load feature arrays.
load('MSCM Files/brodatz-features-train.mat');
brodatzFeatures_train = brodatzFeatures;
clear brodatzFeatures;
load('MSCM Files/brodatz-features-test-30.mat');
brodatzFeatures_test_30 = brodatzFeatures;
clear brodatzFeatures;
load('MSCM Files/brodatz-features-test-60.mat');
brodatzFeatures_test_60 = brodatzFeatures;
clear brodatzFeatures;
load('MSCM Files/brodatz-features-test-90.mat');
brodatzFeatures_test_90 = brodatzFeatures;
clear brodatzFeatures;
load('MSCM Files/brodatz-features-test-120.mat');
brodatzFeatures_test_120 = brodatzFeatures;
clear brodatzFeatures;
load('MSCM Files/brodatz-features-test-150.mat');
brodatzFeatures_test_150 = brodatzFeatures;
clear brodatzFeatures;
load('MSCM Files/brodatz-features-test-200.mat');
brodatzFeatures_test_200 = brodatzFeatures;
clear brodatzFeatures;

%% Create label vector.
labels = [];
for i = 1:13
    for j = 1:16
        labels = cat(1, labels, i);
    end
end
clear i j

%% Create feature vectors for r=1.
x_train = [];
for i = 1:13
    x_train = cat(1, x_train, permute(squeeze(brodatzFeatures_train(:, i, :, 1)), [2 1]));
end, clear i

x_test_30 = [];
for i = 1:13
    x_test_30 = cat(1, x_test_30, permute(squeeze(brodatzFeatures_test_30(:, i, :, 1)), [2 1]));
end, clear i

x_test_60 = [];
for i = 1:13
    x_test_60 = cat(1, x_test_60, permute(squeeze(brodatzFeatures_test_60(:, i, :, 1)), [2 1]));
end, clear i

x_test_90 = [];
for i = 1:13
    x_test_90 = cat(1, x_test_90, permute(squeeze(brodatzFeatures_test_90(:, i, :, 1)), [2 1]));
end, clear i

x_test_120 = [];
for i = 1:13
    x_test_120 = cat(1, x_test_120, permute(squeeze(brodatzFeatures_test_120(:, i, :, 1)), [2 1]));
end

x_test_150 = [];
for i = 1:13
    x_test_150 = cat(1, x_test_150, permute(squeeze(brodatzFeatures_test_150(:, i, :, 1)), [2 1]));
end, clear i

x_test_200 = [];
for i = 1:13
    x_test_200 = cat(1, x_test_200, permute(squeeze(brodatzFeatures_test_200(:, i, :, 1)), [2 1]));
end, clear i

%% Create KNN classifier for r=1.
mdl = fitcknn(x_train, labels, 'NumNeighbors', 1, 'Standardize', false, 'Distance', 'cityblock');

%% Make predictions for r=1.
preds_30 = predict(mdl, x_test_30);
accuracy_30 = sum(preds_30 == labels)/length(labels);

preds_60 = predict(mdl, x_test_60);
accuracy_60 = sum(preds_60 == labels)/length(labels);

preds_90 = predict(mdl, x_test_90);
accuracy_90 = sum(preds_90 == labels)/length(labels);

preds_120 = predict(mdl, x_test_120);
accuracy_120 = sum(preds_120 == labels)/length(labels);

preds_150 = predict(mdl, x_test_150);
accuracy_150 = sum(preds_150 == labels)/length(labels);

preds_200 = predict(mdl, x_test_200);
accuracy_200 = sum(preds_200 == labels)/length(labels);

accuracies_r1 = [accuracy_30 accuracy_60 accuracy_90 accuracy_120 accuracy_150 accuracy_200];
clear x_train x_test_30 x_test_60 x_test_90 x_test_120 x_test_150 ...
      x_test_200 preds_30 preds_60 preds_90 preds_120 preds_150  ...
      preds_200 accuracy_30 accuracy_60 accuracy_90 accuracy_120 ... 
      accuracy_150 accuracy_200 mdl

%% Create feature vectors for r=2.
x_train = [];
for i = 1:13
    x_train = cat(1, x_train, permute(squeeze(brodatzFeatures_train(:, i, :, 2)), [2 1]));
end, clear i

x_test_30 = [];
for i = 1:13
    x_test_30 = cat(1, x_test_30, permute(squeeze(brodatzFeatures_test_30(:, i, :, 2)), [2 1]));
end, clear i

x_test_60 = [];
for i = 1:13
    x_test_60 = cat(1, x_test_60, permute(squeeze(brodatzFeatures_test_60(:, i, :, 2)), [2 1]));
end, clear i

x_test_90 = [];
for i = 1:13
    x_test_90 = cat(1, x_test_90, permute(squeeze(brodatzFeatures_test_90(:, i, :, 2)), [2 1]));
end, clear i

x_test_120 = [];
for i = 1:13
    x_test_120 = cat(1, x_test_120, permute(squeeze(brodatzFeatures_test_120(:, i, :, 2)), [2 1]));
end

x_test_150 = [];
for i = 1:13
    x_test_150 = cat(1, x_test_150, permute(squeeze(brodatzFeatures_test_150(:, i, :, 2)), [2 1]));
end, clear i

x_test_200 = [];
for i = 1:13
    x_test_200 = cat(1, x_test_200, permute(squeeze(brodatzFeatures_test_200(:, i, :, 2)), [2 1]));
end, clear i

%% Create KNN classifier for r=2.
mdl = fitcknn(x_train, labels, 'NumNeighbors', 1, 'Standardize', false, 'Distance', 'cityblock');

%% Make predictions for r=2.
preds_30 = predict(mdl, x_test_30);
accuracy_30 = sum(preds_30 == labels)/length(labels);

preds_60 = predict(mdl, x_test_60);
accuracy_60 = sum(preds_60 == labels)/length(labels);

preds_90 = predict(mdl, x_test_90);
accuracy_90 = sum(preds_90 == labels)/length(labels);

preds_120 = predict(mdl, x_test_120);
accuracy_120 = sum(preds_120 == labels)/length(labels);

preds_150 = predict(mdl, x_test_150);
accuracy_150 = sum(preds_150 == labels)/length(labels);

preds_200 = predict(mdl, x_test_200);
accuracy_200 = sum(preds_200 == labels)/length(labels);

accuracies_r2 = [accuracy_30 accuracy_60 accuracy_90 accuracy_120 accuracy_150 accuracy_200];
clear x_train x_test_30 x_test_60 x_test_90 x_test_120 x_test_150 ...
      x_test_200 preds_30 preds_60 preds_90 preds_120 preds_150  ...
      preds_200 accuracy_30 accuracy_60 accuracy_90 accuracy_120 ... 
      accuracy_150 accuracy_200 mdl

%% Create feature vectors for r=3.
x_train = [];
for i = 1:13
    x_train = cat(1, x_train, permute(squeeze(brodatzFeatures_train(:, i, :, 3)), [2 1]));
end, clear i

x_test_30 = [];
for i = 1:13
    x_test_30 = cat(1, x_test_30, permute(squeeze(brodatzFeatures_test_30(:, i, :, 3)), [2 1]));
end, clear i

x_test_60 = [];
for i = 1:13
    x_test_60 = cat(1, x_test_60, permute(squeeze(brodatzFeatures_test_60(:, i, :, 3)), [2 1]));
end, clear i

x_test_90 = [];
for i = 1:13
    x_test_90 = cat(1, x_test_90, permute(squeeze(brodatzFeatures_test_90(:, i, :, 3)), [2 1]));
end, clear i

x_test_120 = [];
for i = 1:13
    x_test_120 = cat(1, x_test_120, permute(squeeze(brodatzFeatures_test_120(:, i, :, 3)), [2 1]));
end

x_test_150 = [];
for i = 1:13
    x_test_150 = cat(1, x_test_150, permute(squeeze(brodatzFeatures_test_150(:, i, :, 3)), [2 1]));
end, clear i

x_test_200 = [];
for i = 1:13
    x_test_200 = cat(1, x_test_200, permute(squeeze(brodatzFeatures_test_200(:, i, :, 3)), [2 1]));
end, clear i

%% Create KNN classifier for r=3.
mdl = fitcknn(x_train, labels, 'NumNeighbors', 1, 'Standardize', false, 'Distance', 'cityblock');

%% Make predictions for r=3.
preds_30 = predict(mdl, x_test_30);
accuracy_30 = sum(preds_30 == labels)/length(labels);

preds_60 = predict(mdl, x_test_60);
accuracy_60 = sum(preds_60 == labels)/length(labels);

preds_90 = predict(mdl, x_test_90);
accuracy_90 = sum(preds_90 == labels)/length(labels);

preds_120 = predict(mdl, x_test_120);
accuracy_120 = sum(preds_120 == labels)/length(labels);

preds_150 = predict(mdl, x_test_150);
accuracy_150 = sum(preds_150 == labels)/length(labels);

preds_200 = predict(mdl, x_test_200);
accuracy_200 = sum(preds_200 == labels)/length(labels);

accuracies_r3 = [accuracy_30 accuracy_60 accuracy_90 accuracy_120 accuracy_150 accuracy_200];
clear x_train x_test_30 x_test_60 x_test_90 x_test_120 x_test_150 ...
      x_test_200 preds_30 preds_60 preds_90 preds_120 preds_150  ...
      preds_200 accuracy_30 accuracy_60 accuracy_90 accuracy_120 ... 
      accuracy_150 accuracy_200 mdl

%% Create feature vectors for r=4.
x_train = [];
for i = 1:13
    x_train = cat(1, x_train, permute(squeeze(brodatzFeatures_train(:, i, :, 4)), [2 1]));
end, clear i

x_test_30 = [];
for i = 1:13
    x_test_30 = cat(1, x_test_30, permute(squeeze(brodatzFeatures_test_30(:, i, :, 4)), [2 1]));
end, clear i

x_test_60 = [];
for i = 1:13
    x_test_60 = cat(1, x_test_60, permute(squeeze(brodatzFeatures_test_60(:, i, :, 4)), [2 1]));
end, clear i

x_test_90 = [];
for i = 1:13
    x_test_90 = cat(1, x_test_90, permute(squeeze(brodatzFeatures_test_90(:, i, :, 4)), [2 1]));
end, clear i

x_test_120 = [];
for i = 1:13
    x_test_120 = cat(1, x_test_120, permute(squeeze(brodatzFeatures_test_120(:, i, :, 4)), [2 1]));
end

x_test_150 = [];
for i = 1:13
    x_test_150 = cat(1, x_test_150, permute(squeeze(brodatzFeatures_test_150(:, i, :, 4)), [2 1]));
end, clear i

x_test_200 = [];
for i = 1:13
    x_test_200 = cat(1, x_test_200, permute(squeeze(brodatzFeatures_test_200(:, i, :, 4)), [2 1]));
end, clear i

%% Create KNN classifier for r=4.
mdl = fitcknn(x_train, labels, 'NumNeighbors', 1, 'Standardize', false, 'Distance', 'cityblock');

%% Make predictions for r=4.
preds_30 = predict(mdl, x_test_30);
accuracy_30 = sum(preds_30 == labels)/length(labels);

preds_60 = predict(mdl, x_test_60);
accuracy_60 = sum(preds_60 == labels)/length(labels);

preds_90 = predict(mdl, x_test_90);
accuracy_90 = sum(preds_90 == labels)/length(labels);

preds_120 = predict(mdl, x_test_120);
accuracy_120 = sum(preds_120 == labels)/length(labels);

preds_150 = predict(mdl, x_test_150);
accuracy_150 = sum(preds_150 == labels)/length(labels);

preds_200 = predict(mdl, x_test_200);
accuracy_200 = sum(preds_200 == labels)/length(labels);

accuracies_r4 = [accuracy_30 accuracy_60 accuracy_90 accuracy_120 accuracy_150 accuracy_200];
clear x_train x_test_30 x_test_60 x_test_90 x_test_120 x_test_150 ...
      x_test_200 preds_30 preds_60 preds_90 preds_120 preds_150  ...
      preds_200 accuracy_30 accuracy_60 accuracy_90 accuracy_120 ... 
      accuracy_150 accuracy_200 mdl