%% Create MSCMtm
load('MSCM Files/brodatz-train-mscm.mat');
clear Description

%%
%--Image 1 texture group 1 radius 1 = brodaztFeatures(1, 1, 1, :)

brodatzFeatures = zeros(4, 13, 16, 4);

for t = 1:13
    for i = 1:16
        for r = 1:4
            brodatzFeatures(1, t, i, r) = graycoprops(brodatzMSCM_test_60(:, :, t, i, r), 'contrast').Contrast; % Contrast
            brodatzFeatures(2, t, i, r) = graycoprops(brodatzMSCM_test_60(:, :, t, i, r), 'correlation').Correlation; % Correlation
            brodatzFeatures(3, t, i, r) = graycoprops(brodatzMSCM_test_60(:, :, t, i, r), 'energy').Energy; % Energy
            brodatzFeatures(4, t, i, r) = graycoprops(brodatzMSCM_test_60(:, :, t, i, r), 'homogeneity').Homogeneity; % Homogeneity
        end
    end
end

clear t i r

%% save file

save('brodatz-features-test-60.mat', 'brodatzFeatures_test_60');