function corr = correlation_norm(p_ij)
corr = 0;
% Calculating u_x,u_y,s_x,s_y,correlation
u_x=0;
for i=1:32
    for j=1:32
        u_x = u_x + (i.*p_ij(i,j));
    end
end
u_y=0;
for i=1:32
    for j=1:32
        u_y = u_y + (j.*p_ij(i,j));
    end
end
s_x=0;
for i=1:32
    for j=1:32
        s_x = s_x + (((i-u_x)^2).*p_ij(i,j));
    end
end
s_y=0;
for i=1:32
    for j=1:32
        s_y = s_y + (((j-u_y)^2).*p_ij(i,j));
    end
end
corr=0;
for i=1:32
    for j=1:32
        corr = corr + ((i-u_x)*(j-u_y)*p_ij(i,j))/(2*sqrt(s_x*s_y));
    end
end
corr = corr + 0.5;
end
