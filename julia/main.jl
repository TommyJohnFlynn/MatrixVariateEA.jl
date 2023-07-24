include("functions.jl")
using Printf

function ea_mvn(X, ; G=2, K=2, J=4, S=4, tol=1e-6, Z=[], verbose=true)
    total_time = 0
    iter = 0

    # dims
    n, p, N = size(X)

    # Z
    if isempty(Z)
        Z = randStart(N, G, K)
    end

    # Psi 
    Psi_start = Array{Float64}(undef, p, p, G)
    for i in 1:G
        Psi_start[:, :, i] = Matrix(I, p, p)
    end

    # Fit
    fit = Vector{Float64}(undef, K)
    for i in 1:K
        fit[i] = computeFitness(X, Z[:, :, i], Psi_start)["lik"]
    end
    fit, Z = survival(fit, Z, K)
    fit_stag = copy(fit)
    s = 0

    while s < S
        loop_time = time()
        # Crossover
        for k in 1:K
            for j in 1:J
                Z_clone = crossover(Z[:, :, k])
                fit_value = computeFitness(X, Z_clone, Psi_start)["lik"]
                if fit_value > fit[k]
                    Z = cat(Z, Z_clone, dims=3)
                    fit = vcat(fit, fit_value)
                end
            end
        end
        fit, Z = survival(fit, Z, K)

        # Mutation 
        for k in 1:K
            m = mutate(Z[:, :, K], fit[K], X, Psi_start)
            if m["fl"]
                Z = cat(Z, m["Z"], dims=3)
                fit = vcat(fit, m["f"])
                Psi_start = m["P"]
            end
        end
        fit, Z = survival(fit, Z, K)

        # Stagnation
        if isapprox(fit_stag, fit; rtol=tol)
            s += 1
        else
            s = 0
        end
        fit_stag = copy(fit)
        if verbose
            iter += 1
            dloop_time = time() - loop_time
            total_time += dloop_time
            println("Iter: \u001b[32m", iter, "\u001b[0m | Stag: \u001b[33m", s, "\u001b[0m | Fit: \u001b[34m", @sprintf("%.2f", round(fit[1], digits=2)),
                "\u001b[0m | Loop time: \u001b[36m", @sprintf("%.2f", round(dloop_time, digits=2)), "\u001b[0m | Total time: \u001b[31m", @sprintf("%.2f", round(total_time, digits=2)), "\u001b[0m")
        end
    end

    # Best assignment 
    ea_map = Vector{Int}(undef, N)
    for i in 1:N
        ea_map[i] = findfirst(x -> x == 1, Z[i, :, 1])
    end
    return Dict("Z" => Z, "fit" => fit, "class" => ea_map)
end

