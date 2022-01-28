function eng = energy_norm(p_ij)
eng = 0;
for i=1:32
    for j=1:32
        eng = eng + p_ij(i,j)^2;
    end
end
end
