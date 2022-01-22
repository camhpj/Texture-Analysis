%% Create MSCMtm
x = 'train';

load('MSCM Files/brodatz-mscm-' + x + '.mat');
clear Description

%%
%--Image 1 texture group 1 radius 1 = brodaztFeatures(1, 1, 1, :)

brodatzFeatures_train = zeros(4, 13, 16, 4);

for t = 1:13
    for i = 1:16
        for r = 1:4
            brodatzFeatures_train(1, t, i, r) = graycoprops(brodatzMSCM_train(:, :, t, i, r), 'contrast').Contrast; % Contrast
            brodatzFeatures_train(2, t, i, r) = graycoprops(brodatzMSCM_train(:, :, t, i, r), 'correlation').Correlation; % Correlation
            brodatzFeatures_train(3, t, i, r) = graycoprops(brodatzMSCM_train(:, :, t, i, r), 'energy').Energy; % Energy
            brodatzFeatures_train(4, t, i, r) = graycoprops(brodatzMSCM_train(:, :, t, i, r), 'homogeneity').Homogeneity; % Homogeneity
            %brodatzFeatures(5, t, i, r) = -sum(sum((MSCM.*log(brodatzMSCM_test_60(:, :, t, i, r) + eps)))); % Entropy
        end
    end
end

clear t i r

%% save file
save('MSCM Files/brodatz-features-' + x + '.mat', 'brodatzFeatures_train');