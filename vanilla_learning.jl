using Stheno, Zygote, Plots, Random, Optim
using Stheno: @model

@model function model(θ)
    _, l1, w1, _, l2, w2 = exp.(θ)

    f1 = w1 * stretch(GP(eq()), l1) - 5.0
    f2 = w2 * stretch(GP(eq()), l2)
    f3 = f1 + f2 + 10.0
    return f1, f2, f3
end

rng = MersenneTwister(123456);
xp = collect(range(-15.0, 15.0; length=1000));
x1, x3 = 5 .* randn(rng, 75), 5 .* randn(rng, 125);

σ²1, σ²3 = 1e-1, 1e-2;
f1, f2, f3 = model(log.([σ²1, 0.5, 1.5, σ²3, 1.5, 0.5]));
y1, y3, y1_true, y2_true, y3_true = rand(
    rng,
    [f1(x1, σ²1 + 1e-6), f3(x3, σ²3 + 1e-6), f1(xp, 1e-6), f2(xp, 1e-6), f3(xp, 1e-6)],
);

function nlml(θ)
    f1, f2, f3 = model(θ)
    fx1, fx3 = f1(x1, exp(θ[1]) + 1e-6), f3(x3, exp(θ[3]) + 1e-6)
    return -logpdf(fx1 ← y1, fx3 ← y3)
end

# Initialise at the truth to see what the optimal solution looks like
θ0 = log.([σ²1, 0.5, 1.5, σ²3, 1.5, 0.5]);

# Use quasi 2nd-order optimisation to find optimal hyperparameters
options = Optim.Options(show_trace=true, iterations=25);
results = optimize(
    nlml,
    θ->first(Zygote.gradient(nlml, θ)),
    copy(θ0),
    BFGS(),
    options;
    inplace=false,
)
θ_opt = results.minimizer;

# Sample from infered posterior
let
    rng, S = MersenneTwister(123456), 25

    # Construct posterior processes given optimal parameters
    f1, f2, f3 = model(θ_opt)
    σ²1, σ²3 = exp(θ_opt[1]), exp(θ_opt[4])
    (f1′, f2′, f3′) = (f1, f2, f3) | (f1(x1, σ²1) ← y1, f3(x3, σ²3) ← y3)
    f1′xp, f2′xp, f3′xp = f1′(xp, 1e-9), f2′(xp, 1e-9), f3′(xp, 1e-9)

    # Generate posterior samples
    f1′xp_samples, f2′xp_samples, f3′xp_samples = rand(rng, [f1′xp, f2′xp, f3′xp], S)

    # Compute posterior marginal statistics
    ms1 = marginals(f1′xp)
    ms2 = marginals(f2′xp)
    ms3 = marginals(f3′xp)

    m1, σ1 = mean.(ms1), std.(ms1)
    m2, σ2 = mean.(ms2), std.(ms2)
    m3, σ3 = mean.(ms3), std.(ms3)

    # Visualise the posterior
    pyplot()
    plot_dir = joinpath("intro_to_gps", "figs", "vanilla_learning")
    plt = plot()

    # Plot f1
    plot!(plt, xp, f1′xp_samples; linecolor=:red, label="", linealpha=0.2)
    plot!(plt, xp, y1_true; linecolor=:red, label="f1", linestyle=:dash, linewidth=2)
    plot!(plt, xp, [m1 m1];
        linewidth=0.0,
        fillrange=[m1 .- 3 .* σ1, m1 .+ 3 * σ1],
        fillalpha=0.3,
        fillcolor=:red,
        label="",
    )

    # Plot f2
    plot!(plt, xp, f2′xp_samples; linecolor=:green, label="", linealpha=0.2)
    plot!(plt, xp, y2_true; linecolor=:green, label="f2", linestyle=:dash, linewidth=2)
    plot!(plt, xp, [m2 m2];
        linewidth=0.0,
        fillrange=[m2 .- 3 .* σ2, m2 .+ 3 * σ2],
        fillalpha=0.3,
        fillcolor=:green,
        label="",
    )

    # Plot f3
    plot!(plt, xp, f3′xp_samples; linecolor=:blue, label="", linealpha=0.2)
    plot!(plt, xp, y3_true; linecolor=:blue, label="f3", linestyle=:dash, linewidth=2)
    plot!(plt, xp, [m3 m3];
        linewidth=0.0,
        fillrange=[m3 .- 3 .* σ3, m3 .+ 3 * σ3],
        fillalpha=0.3,
        fillcolor=:blue,
        label="",
    )

    scatter!(plt, x1, y1;
        label="",
        markershape=:x,
        markercolor=:red,
        markersize=6,
        markeralpha=1,
        markerstrokewidth=0.8,
    )
    scatter!(plt, x3, y3;
        label="",
        markershape=:x,
        markercolor=:blue,
        markersize=6,
        markeralpha=1,
        markerstrokewidth=0.8,
    )
    savefig(plt, joinpath(plot_dir, "all.pdf"))
end
