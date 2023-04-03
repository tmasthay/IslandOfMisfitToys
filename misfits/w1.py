import numpy as np

#def func1():
#    print('func1')


function W1_metric(f, g, t, Nt, dt)
# normalization
    f = f ./ (norm(f,1)*dt);
    g = g ./ (norm(g,1)*dt);
# numerical integration
    F = zeros(Nt);
    G = zeros(Nt);
    for i = 1:Nt
        F[i] = sum(f[1:i])
        G[i] = sum(g[1:i])
    end
    F = F .* dt;
    G = G .* dt;
# inverse of G 
    G_inv = zeros(Nt);
    for i = 1:Nt
        y = F[i]
        ind_g = findall(x -> x >= y, G)
        if length(ind_g) == 0
            G_inv[i] = t[end];
        else
            G_inv[i] = t[ind_g[1]];
        end
    end
    w1 = sum((t-G_inv) .* f * dt)
    return w1
end
