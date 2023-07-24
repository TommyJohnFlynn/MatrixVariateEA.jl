using Random
using LinearAlgebra
using Clustering

function randStart(N, G, K)
    result = zeros(Int, N, G, K)
    for k in 1:K
        for i in 1:N
            j = rand(1:G)
            result[i, j, k] = 1
        end
    end
    return result
end

function MVN(X, M, Sigmain, Psiin, n, p, detSig, detPsi)
    logdensNorm = -(n * p / 2) * log(2 * pi) + (p / 2) * log(detSig) + (n / 2) * log(detPsi) - 0.5 * tr(Sigmain * (X - M) * Psiin * (X - M)')
    return exp(logdensNorm)
end


function update(X, Z, Psi_start)
    n, p, N = size(X)
    G = size(Z, 2)
    M = zeros(Float64, n, p, G)
    Sigma = zeros(Float64, n, n, G)
    Psi = zeros(Float64, p, p, G)
    N_g = sum(Z, dims=(1))
    Pi = N_g / N
    Psi_g_inv = [inv(Psi_start[:, :, g]) for g in 1:G]
    for g in 1:G
        non_zeros = findall(Z[:, g] .== 1)
        M_g = sum(X[:, :, i] for i in non_zeros) / N_g[g]
        N_non_zeros = length(non_zeros)
        A_g = [X[:, :, i] - M_g for i in non_zeros]
        Sigma_g = sum(A_g[i] * Psi_g_inv[g] * transpose(A_g[i]) for i in 1:N_non_zeros) / (p * N_g[g])
        Sigma_g_inv = inv(Sigma_g)
        Psi_g = sum(transpose(A_g[i]) * Sigma_g_inv * A_g[i] for i in 1:N_non_zeros) / (n * N_g[g])
        M[:, :, g] = M_g
        Sigma[:, :, g] = Sigma_g
        Psi[:, :, g] = Psi_g
    end
    return Dict("M" => M, "Sigma" => Sigma, "Psi" => Psi, "Pi" => Pi)
end

function obsLik(X, M, Sigma, Psi, Pi)
    n, p, N = size(X)
    G = size(M)[3]
    log_lik = 0.0
    Sigmainv = [inv(Sigma[:, :, g]) for g in 1:G]
    Psiinv = [inv(Psi[:, :, g]) for g in 1:G]
    detSig = [det(Sigmainv[g]) for g in 1:G]
    detPsi = [det(Psiinv[g]) for g in 1:G]
    for i in 1:N
        count_g = 0.0
        for g in 1:G
            count_g += Pi[g] * MVN(X[:, :, i], M[:, :, g], Sigmainv[g], Psiinv[g], n, p, detSig[g], detPsi[g])
        end

        log_lik += log(count_g)
    end
    return log_lik
end

function computeFitness(X, Z, Psi_start)
    e = update(X, Z, Psi_start)
    lik = obsLik(X, e["M"], e["Sigma"], e["Psi"], e["Pi"])
    return Dict("lik" => lik, "M" => e["M"], "Sigma" => e["Sigma"], "Psi" => e["Psi"], "Pi" => e["Pi"])
end

function crossover(Z)
    N = size(Z, 1)
    row_list = randperm(N)
    i = row_list[1]
    i_index = findfirst(Z[i, :] .== 1)
    for j in row_list[2:N]
        j_index = findfirst(Z[j, :] .== 1)
        if i_index != j_index
            Z[i, :], Z[j, :] = Z[j, :], Z[i, :]
            return Z
        end
    end
    return Z
end


function mutate(Z, f, X, Psi_start)
    N, G = size(Z)
    perm = randperm(N)
    Z_temp = similar(Z)
    for r in perm
        Z_temp = copy(Z)
        one_index = findfirst(x -> x == 1, Z[r, :])
        zero_index = rand(setdiff(1:G, one_index))
        Z_temp[r, one_index] = 0
        Z_temp[r, zero_index] = 1
        c = computeFitness(X, Z_temp, Psi_start)
        f_temp = c["lik"]
        if f_temp > f
            return Dict("Z" => Z_temp, "f" => f_temp, "P" => c["Psi"], "fl" => true)
        end
    end
    return Dict("Z" => Z, "f" => f, "fl" => false)
end

function survival(fit, Z, K)
    sorted_indices = partialsortperm(fit, 1:K, rev=true)
    return fit[sorted_indices], Z[:, :, sorted_indices]
end

function BIC_Calc(lik, n, p, N, G)
    npar = G - 1 + G * (n * p + n * (n + 1) / 2 + p * (p + 1) / 2) - G
    BIC = 2 * maximum(lik) - log(N) * npar
    return BIC
end
