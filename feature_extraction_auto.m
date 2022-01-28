%%
brodatzFeatures = zeros(4, 13, 16, 4);
orientations = ["train" "test-30" "test-60" "test-90" "test-120" "test-150" "test-200"];

for o = 1:7
    load_name = strcat('MSCM Files/brodatz-mscm-', orientations(o), '.mat');
    load(load_name);
    for t = 1:13
        for i = 1:16
            for r = 1:4
                norm = sum(sum(brodatzMSCM(:, :, t, i, r)));
                MSCM = brodatzMSCM(:, :, t, i, r) ./ norm;
                brodatzFeatures(1, t, i, r) = contrast_norm(MSCM);
                brodatzFeatures(2, t, i, r) = correlation_norm(MSCM);
                brodatzFeatures(3, t, i, r) = energy_norm(MSCM);
                brodatzFeatures(4, t, i, r) = homogeneity_norm(MSCM);
            end
        end
    end
    
    save_name = strcat('MSCM Files/brodatz-features-', orientations(o), '.mat');
    save(save_name, 'brodatzFeatures');
end
