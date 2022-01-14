%% Create MSCMtm
load('MSCM Files/brodatz-mscm-test-200.mat');
clear Description

%%
%--Image 1 texture group 1 radius 1 = brodaztFeatures(1, 1, 1, :)

brodatzFeatures_test_200 = zeros(4, 13, 16, 4);

for t = 1:13
    for i = 1:16
        for r = 1:4
            brodatzFeatures_test_200(1, t, i, r) = graycoprops(brodatzMSCM_test_200(:, :, t, i, r), 'contrast').Contrast; % Contrast
            brodatzFeatures_test_200(2, t, i, r) = graycoprops(brodatzMSCM_test_200(:, :, t, i, r), 'correlation').Correlation; % Correlation
            brodatzFeatures_test_200(3, t, i, r) = graycoprops(brodatzMSCM_test_200(:, :, t, i, r), 'energy').Energy; % Energy
            brodatzFeatures_test_200(4, t, i, r) = graycoprops(brodatzMSCM_test_200(:, :, t, i, r), 'homogeneity').Homogeneity; % Homogeneity
            %brodatzFeatures(5, t, i, r) = -sum(sum((MSCM.*log(brodatzMSCM_test_60(:, :, t, i, r) + eps)))); % Entropy
        end
    end
end

clear t i r

%% save file
save('MSCM Files/brodatz-features-test-200.mat', 'brodatzFeatures_test_200');