
using Revise

using Turing, Stheno, Plots, Flux, LinearAlgebra
using Stheno: TuringGPC
using Turing: @model
using LinearAlgebra: AbstractTriangular

Turing.setadbackend(:reverse_diff)

#
# Define extra methods for Tracker
#

import Base: *, \
function *(A::Adjoint{T, <:AbstractTriangular{T}} where {T<:Real}, x::TrackedVector{<:Real})
    return Tracker.track(*, A, x)
end
function *(A::Transpose{T, <:AbstractTriangular{T}} where {T<:Real}, x::TrackedVector{<:Real})
    return Tracker.track(*, A, x)
end
*(A::AbstractTriangular{<:Real}, x::TrackedVector{<:Real}) = Tracker.track(*, A, x)

function \(
    A::Adjoint{T, <:Union{LowerTriangular{T}, UpperTriangular{T}}} where {T<:Real},
    x::TrackedVector{<:Real},
)
    return Tracker.track(\, A, x)
end
*(A::TrackedMatrix{<:Real}, X::AbstractTriangular{<:Real}) = Tracker.track(*, A, x)



#
# Define a vaguely interesting non-Gaussian regression problem
#

N = 50;
x = collect(range(-5.0, 5.0; length=N));
# x_pr = collect(range(-5.0, 5.0; length=25));

@model gaussian_regression(y) = begin
    f = GP(eq(), TuringGPC())
    fx ~ f(x, 1e-9)
    # fx_pr ~ f(x_pr, 1e-1)
    y ~ Product(Normal.(fx, 0.1))
    return fx, y
end



#
# Sample from the prior and plot it
#

fx, y = gaussian_regression()();

plotly();
plot(x, fx; label="Latent at observations");
scatter!(x, y; label="Observations")
# plot!(x_pr, fx_pr; label="")



#
# Do posterior inference via HMC (NUTS)
#

N, N_burn, N_thin = 1_500, 500, 10;
chain = sample(gaussian_regression(y), NUTS(N, N_burn, 0.2));

fx_slices = get(chain, :fx)[1];
fx_samples = Float64.(hcat(fx_slices...));

# fx_pr_slices = get(chain, :fx_pr)[1];
# fx_pr_samples = Float64.(hcat(fx_pr_slices...));

plot(x, fx_samples[N_burn+1:N_thin:end, :]'; label="", linecolor=:red);
scatter!(x, y);
plot!(x, fx; linecolor=:blue)
# plot!(x_pr, fx_pr_samples[N_burn+1:N_thin:end, :]'; linecolor=:green, label="")




#
# Logistic regression equivalent example
#

x = collect(range(-5.0, 5.0; length=50));

@model logistic_regression(y) = begin
    f = 5.0 * GP(eq(), TuringGPC())
    fx ~ f(x, 1e-9)
    y ~ Product(Bernoulli.(σ.(fx)))
    return fx, y
end

fx, y = logistic_regression()();

N, N_burn, N_thin = 2_500, 500, 10;
chain = sample(logistic_regression(y), NUTS(N, N_burn, 0.25));
fx_slices = first(get(chain, :fx));
fx_samples = Float64.(hcat(fx_slices...))[N_burn+1:N_thin:end, :]';

plot(x, fx_samples; label="", linecolor=:red);
scatter!(x, y);
plot!(x, fx; linecolor=:blue)


#
# Poisson regression
#

x = range(-5.0, 5.0; length=50);

@model poisson_regression(y) = begin
    f = GP(eq(), TuringGPC())
    fx ~ f(x, 1e-9)
    y ~ Product(Poisson.(exp.(fx)))
    return fx, y
end

fx, y = poisson_regression()();

N, N_burn, N_thin = 5_000, 4_000, 10;
chain = sample(poisson_regression(y), HMC(N, 3e-3, 15));

let
    fx_slices = get(chain, :fx)[1];
    fx_samples_full = Float64.(hcat(fx_slices...))[N_burn+1:end, :]'
    fx_samples = fx_samples_full[:, 1:N_thin:end];

    pyplot()
    plt = plot()
    plot!(x, fx_samples;
        label="",
        linecolor=:blue,
        linealpha=0.2,
        linewidth=0.7,
    )
    plot!(x, fx;
        label="",
        linecolor=:blue,
        linestyle=:dash,
        linewidth=2.0,
    )

    m, σ = vec(mean(fx_samples_full; dims=2)), vec(std(fx_samples_full; dims=2))
    plot!(x, m;
        label="",
        linecolor=:blue,
        linealpha=1.0,
        linewidth=2.0,
    )
    plot!(plt, x, [m m];
        linewidth=0.0,
        fillrange=[m .- 3 .* σ, m .+ 3 * σ],
        fillalpha=0.3,
        fillcolor=:blue,
        label="",
    )

    scatter!(plt, x, y;
        label="",
        markershape=:x,
        markercolor=:black,
        markersize=6,
        markeralpha=1,
        markerstrokewidth=0.8,
    )
    savefig(plt, joinpath("intro_to_gps", "figs", "poisson_regression_posterior.pdf"))
end



    # g′ = g | (f₁(x₁) ← y₁, f₂(x₂) ← y₂)


    # rand(rng, [f₁(x₁), f₂(x₂)], N_samples)


    # logpdf([f₁(x₁), f₂(x₂)], [y₁, y₂])





#
# Heteroscedastic Gaussian processes - only sort of works
#

x = range(-5.0, 5.0; length=50);

@model heteroscedastic_regresion(y) = begin

    # Specify distribution over noise
    logσx ~ (2 * GP(eq(l=0.5), TuringGPC()))(x, 1e-9)

    # Specify latent GP
    fx ~ GP(eq(l=2.0), TuringGPC())(x, 1e-9)

    # Generate observations
    y ~ Product(Normal.(softplus.(logσx) .* fx, 0.1))

    return logσx, fx, y
end

# Generate from the prior
logσx, fx, y = heteroscedastic_regresion()()

# Plot the samples from the prior
let
    pyplot()
    plt = plot()
    σ = softplus.(logσx)
    plot!(x, σ; linecolor=:blue, label="process std")
    plot!(x, fx; linecolor=:red, label="latent function")
    plot!(x, σ .* fx; linecolor=:green, label="function")
    scatter!(x, y; markercolor=:green)
    display(plt)
end

# Perform posterior inference
N, N_burn, N_thin = 1_000, 1_50, 10;
chain = sample(heteroscedastic_regresion(y), HMC(N, 1e-3, 10));

# Plot the results of posterior inference
let
    fx_slices = first(get(chain, :fx))
    fx_samples = Float64.(hcat(fx_slices...))[N_burn+1:N_thin:end, :]'

    logσx_slices = first(get(chain, :logσx))
    logσx_samples = softplus.(Float64.(hcat(logσx_slices...))[N_burn+1:N_thin:end, :]')

    pyplot()
    plt = plot()

    plot!(plt, x, logσx_samples[:, 1]; label="σ", linecolor=:red, alpha=0.2)
    plot!(plt, x, logσx_samples; label="", linecolor=:red, alpha=0.2)
    plot!(plt, x, softplus.(logσx);
        label="",
        linecolor=:red,
        linestyle=:dash,
        linewidth=2.0,
    )

    # plot!(plt, x, fx_samples[:, 1]; label="f", linecolor=:green, alpha=0.2)
    # plot!(plt, x, fx_samples; label="", linecolor=:green, alpha=0.2)
    # plot!(plt, x, fx;
    #     label="",
    #     linecolor=:green,
    #     linestyle=:dash,
    #     linewidth=2.0,
    # )

    plot!(plt, x, logσx_samples[:, 1] .* fx_samples[:, 1];
        label="σ * f",
        linecolor=:blue,
        alpha=0.2,
    )
    plot!(plt, x, logσx_samples .* fx_samples;
        label="",
        linecolor=:blue,
        alpha=0.2,
    )
    plot!(plt, x, softplus.(logσx) .* fx;
        label="",
        linecolor=:blue,
        linestyle=:dash,
        linewidth=2.0,
    )

    scatter!(plt, x, y;
        label="observations",
        markershape=:x,
        markercolor=:black,
        markersize=6,
        markeralpha=1,
        markerstrokewidth=0.8,
    )
end


function jacobian(b::ADBijector{<: ForwardDiffAD}, y::Real)
    ForwardDiff.derivative(z -> transform(b, z), y)
end
