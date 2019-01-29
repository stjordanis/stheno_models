using Stheno, Plots, Random, Statistics
using Stheno: @model

# Sample a few times from the prior.
@model model(α, β, l) = GP(eq(α=α, β=β, l=l));
f = model(1, 15, 1.3);
x = range(-3.0, stop=3.0, length=1000);
scatter(x, rand(f(x, 1e-3), 10);
    markersize=0.1,
    markershape=:xcross,
    markerstrokewidth=0.0,
)

# Compute the logpdf of a sample.
σ² = 1e-3;
y = rand(f(x, σ²));
logpdf(f(x, σ²), y)

# Compute + plot the posterior distribution over the process.
f′ = f | (f(x, σ²) ← y);

x_plot = range(-4.0, stop=4.0, length=1000);
f′_xp = f′(x_plot, 1e-6);
μ′_xp, σ′_xp = mean.(marginals(f′_xp)), std.(marginals(f′_xp));

plt = plot(legend=nothing);
scatter!(plt, x, y;
    markersize=0.1,
    markershape=:xcross,
    markerstrokewidth=0.0,
);
plot!(plt, x_plot, mean(f′_xp);
    linewidth=2,
    linecolor=:blue,
);
# plot!(plt, x_plot, mean(f′_xp) .+ 3 .* std.(marginals(f′_xp)));
plot!(plt, x_plot, [μ′_xp μ′_xp];
    linewidth=0.0,
    linecolor=:blue,
    fillrange=[μ′_xp .- 3 .* σ′_xp, μ′_xp .+ 3 * σ′_xp],
    fillalpha=0.3,
    fillcolor=:blue,
);

# Compute posterior distribution over _observations_.
y′_xp = f′(x_plot, σ²);
μ′_xp, σ′_xp = mean.(marginals(y′_xp)), std.(marginals(y′_xp));
plot!(plt, x_plot, [μ′_xp μ′_xp];
    linewidth=0.0,
    linecolor=:blue,
    fillrange=[μ′_xp .- 3 .* σ′_xp, μ′_xp .+ 3 * σ′_xp],
    fillalpha=0.3,
    fillcolor=:red,
);

display(plt);
