function hom = homogeneity_norm(p_ij)
hom = 0;
for i=1:32
    for j=1:32
        hom = hom + (p_ij(i,j))/(1+abs(i-j));
    end
end
