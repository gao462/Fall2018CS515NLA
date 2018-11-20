using DelimitedFiles
using LinearAlgebra

function analyze(dtype, epsilon)
    candyland_matrix = readdlm("candyland-matrix.csv", ',')
    A = zeros(dtype, 140, 140)
    for i = 1 : size(candyland_matrix, 1)
        row = Int(candyland_matrix[i, 1])
        col = Int(candyland_matrix[i, 2])
        val = candyland_matrix[i, 3]
        A[row, col] = val
    end
    # b = ones(dtype, 140, 1)
    # b[134, 1] = 0
    # println((inv(I - transpose(A)) * b)[140, 1])
    b = zeros(dtype, 140, 1)
    b[140, 1] = 1
    k = 1
    p = A * b
    S = k * p
    while true
        k += 1
        p = A * p
        if k * norm(p) < epsilon
            break
        else
            S = S + k * p
        end
    end
    println(S[134, 1])
end

analyze(Float16, 1e-4)
analyze(BigFloat, 1e-16)