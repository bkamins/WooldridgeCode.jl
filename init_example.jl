# Defined functions:
# - get_dataset
# - ser
# - hvcov
# - lm_ttest
# - lm_waldtest
# - awts
# - lad
# - studentized_residuals

using DataFramesMeta
using Distributions
using FreqTables
using GLM
using LinearAlgebra
using Optim
using Plots
using Statistics
using StatsBase

import CodecXz
import Downloads
import RData

function get_dataset(name::AbstractString)
    filename = name * ".RData"
    if !isfile(filename)
        Downloads.download("https://github.com/JustinMShea/wooldridge/" *
                           "raw/master/data/$name.RData", filename)
    end
    ds = RData.load(filename)
    return ds[name]
end

ser(model) = sqrt(deviance(model) / dof_residual(model))

function hvcov(model, type::Symbol=:SE)
    @assert type in [:SE, :HC0, :HC1, :HC2, :HC3]
    @assert model.mf.model == LinearModel

    X = modelmatrix(model)
    u² = residuals(model) .^ 2

    wts = model.model.rr.wts

    if !isempty(wts)
        # assume properly scaled analytical weights
        @assert length(wts) == sum(wts)
        X = X .* sqrt.(wts)
        u² .*= wts
    end

    n = nobs(model)
    dfr = dof_residual(model)
    M = inv(X'*X)

    if type == :SE
        return M * sum(u²) / dfr
    end

    if type == :HC0 || type == :HC1
        D = Diagonal(u²)
        P = X' * D * X
        HC0 = M * P * M
        if type == :HC1
            return (n / dfr) * HC0
        else
            return HC0
        end
    end

    if type == :HC2 || type == :HC3
        h = [x' * M * x for x in eachrow(X)]
        if type == :HC2
            D2 = Diagonal(u² ./ (1.0 .- h))
            P2 = X' * D2 * X
            return M * P2 * M
        end
        if type == :HC3
            D3 = Diagonal(u² ./ (1.0 .- h) .^ 2)
            P3 = X' * D3 * X
            return M * P3 * M
        end
    end
end

function lm_ttest(model, type::Symbol=:SE)
    dfr = dof_residual(model)
    V = hvcov(model, type)
    se = sqrt.(diag(V))
    t = coef(model) ./ se
    return DataFrame("variable" => coefnames(model),
                     "Coef." => coef(model),
                     "Std. Error" => se,
                     "t" => t,
                     "Pr(>|t|))" => StatsBase.PValue.(2 .* ccdf.(TDist(dfr), abs.(t))))
end

function lm_waldtest(model, R::AbstractMatrix, r::AbstractVector, type::Symbol=:SE)
    β = coef(model)
    N = nobs(model)
    P = length(β)
    Q = length(r)
    D = R * β - r
    V = hvcov(model, type)

    @assert (Q, P) == size(R)

    W = D' * inv(R * V * R') * D / Q
    pvalue = ccdf(FDist(Q, N - P), W)
    return (W=W, pvalue=StatsBase.PValue.(pvalue))
end

# TODO: fix this after https://github.com/JuliaStats/GLM.jl/pull/487 is merged
awts(x) = Float64.((big(1) ./ x) ./ mean(big(1) ./ x))

function lad(model)
    @assert model.mf.model == LinearModel
    X = modelmatrix(model)
    y = model.model.rr.y
    β = coef(model)
    wts = model.model.rr.wts
    if isempty(wts)
        wts = push!(copy(wts), 1.0)
    end
    res = optimize(β, NelderMead()) do beta
        sum(abs, (y .- X * beta) .* wts)
    end
    if Optim.converged(res)
        @info "LAD optimization successful. Objective: $(Optim.minimum(res))"
    else
        @error "LAD optimization failed to converge. Objective: $(Optim.minimum(res))"
    end
    return Optim.minimizer(res)
end

function studentized_residuals(model)
    @assert model.mf.model == LinearModel
    @assert isempty(model.model.rr.wts)

    X = modelmatrix(model)
    h = diag(X * inv(X' * X) * X')
    return residuals(model) ./ (ser(model) * sqrt.(1.0 .- h))
end
