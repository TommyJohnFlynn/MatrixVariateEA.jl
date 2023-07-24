using DataFrames
using Feather
using StatsBase
include("main.jl")
Random.seed!(21)

# set-up
G = 2
K = 1
J = 12
S = 3

# data
file_name = "example.feather"
df = Feather.read(file_name)
true_class = df[:, 1]
true_class = convert.(Int, true_class)
X = Array{Float64}(undef, 3, 4, nrow(df))
for i in 1:nrow(df)
    original_matrix = reshape(hcat(df[i, 2:13]...), 3, 4)
    X[:, :, i] = original_matrix
end
n, p, N = size(X)

@time begin
    EA = ea_mvn(X, G=G, K=K, J=J, S=S, verbose=true)
end

# BIC
println("BIC: ", BIC_Calc(EA["fit"][1], n, p, N, G))
# ARI
println("ARI: ", randindex(true_class, EA["class"])[1])
