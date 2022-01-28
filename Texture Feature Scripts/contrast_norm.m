% Calculating Contrast proposed in DFT Paper
function con = contrast_norm(p_ij)
con=0;
for i=1:32
    for j=1:32
        con = con + ((abs(i-j)^2)*p_ij(i,j))/32^2;
    end
end
end