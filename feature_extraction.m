load('MSCM Files/brodatz-mscm-test-200.mat');
clear Description

%% Calculate and construct feature matrix.
%--brodatzFeatures(numFeature, numTexture, numImage, radius)
%--contrast=1, correlation=2, energy=3, homogeneity=4

numTextures = 0;
numImages = 0;
numRadii = 0;
fileName = "brodatzFeatures_{test or train}_{angle}.mat";

brodatzFeatures = zeros(4, numTextures, numImages, numRadii);

for t = 1:numTextures
    for i = 1:numImages
        for r = 1:numRadii
            brodatzFeatures(1, t, i, r) = graycoprops(brodatzMSCM(:, :, t, i, r), 'contrast').Contrast;       % Contrast
            brodatzFeatures(2, t, i, r) = graycoprops(brodatzMSCM(:, :, t, i, r), 'correlation').Correlation; % Correlation
            brodatzFeatures(3, t, i, r) = graycoprops(brodatzMSCM(:, :, t, i, r), 'energy').Energy;           % Energy
            brodatzFeatures(4, t, i, r) = graycoprops(brodatzMSCM(:, :, t, i, r), 'homogeneity').Homogeneity; % Homogeneity
        end
    end
end

clear t i r 

%% Save file.
save(sprintf('MSCM Files/%f', fileName), 'brodatzFeatures');