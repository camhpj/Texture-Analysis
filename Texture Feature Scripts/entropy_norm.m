function en = entropy_norm(p_ij)
en = 0;
for i=1:32
    for j=1:32
        if p_ij(i,j) > 0
        en = en + (p_ij(i,j)*log2(p_ij(i,j)))/(-(2*log2(32)));
        end
    end
end