using Stheno, Plots, Random, FileIO, LatexPrint
using Stheno: GPC
pyplot();

plot_dir = joinpath("intro_to_gps", "figs");

# Create a very simple GP.
f = GP(0, eq(), GPC());

# Increasing numbers of points with same width.
let
    N, S = 11, 5;
    ys = Vector{Vector{Float64}}(undef, S)
    let
        rng = MersenneTwister(123456);

        # Generate initial samples to condition on.
        x = collect(0:0)
        fs = rand(rng, f(x, 1e-12), S)
        for s in 1:S
            ys[s] = [fs[s]]
        end

        # Plot samples.
        plt = plot(xlim=[0, N-1], ylim=[-3, 3], xticks=0:N-1)

        savefig(plt, joinpath(plot_dir, "range_samples_0.pdf"))

        # Save covariance matrix
        write(
            joinpath(plot_dir, "cov_range_samples_0.tex"),
            latex_form(round.(cov(f(x)); digits=2)),
        )

        for n in 1:N-1

            # Generate samples.
            x = collect(0:n)
            for s in 1:S
                y_new = rand(rng, (f | (f(x[1:end-1], 1e-12) ← ys[s]))(x, 1e-12))
                ys[s] = y_new
            end

            # Plot samples.
            plt = plot(xlim=[0, N-1], ylim=[-3, 3], xticks=0:N-1)
            plot!(plt, x, hcat(ys...);
                marker=:x,
                label="",
                linecolor=:blue,
                markercolor=:blue,
            )
            savefig(plt, joinpath(plot_dir, "range_samples_$n.pdf"))

            # Save covariance matrix
            write(
                joinpath(plot_dir, "cov_range_samples_$n.tex"),
                latex_form(round.(cov(f(x)); digits=2)),
            )
        end
    end

    # Plot points with increased covariance
    let
        f = GP(0, eq(l=0.3), GPC())
        x = collect(0:N-1)
        fs = rand(rng, f(x, 1e-12), S)
        plt = plot(xlim=[0, N-1], ylim=[-3, 3], xticks=0:N-1)
        plot!(plt, x, fs;
            marker=:x,
            label="",
            linecolor=:blue,
            markercolor=:blue,
        )
        savefig(plt, joinpath(plot_dir, "range_samples_length_3_10.pdf"))
    end

    # Plot points with decreased covariance
    let
        f = GP(0, eq(l=2), GPC())
        x = collect(0:N-1)
        fs = rand(rng, f(x, 1e-12), S)
        plt = plot(xlim=[0, N-1], ylim=[-3, 3], ticks=0:N-1)
        plot!(plt, x, fs;
            marker=:x,
            label="",
            linecolor=:blue,
            markercolor=:blue,
        )
        savefig(plt, joinpath(plot_dir, "range_samples_length_2.pdf"))
    end

    # Increase density of points with same width
    rng, D = MersenneTwister(123456), 5;
    x_prev = collect(range(0, N-1; length=N))
    for d in 0:D

        # Generate samples.
        # x = collect(range(0, N-1; length=N * d))
        x = collect(0.0:1 / 2^d:float(N-1))
        for s in 1:S
            ys[s] = rand(rng, (f | (f(x_prev, 1e-12) ← ys[s]))(x, 1e-12))
        end

        # Plot samples with lines.
        plt = plot(xlim=[0.0, N-1], ylim=[-3, 3], xticks=0:N-1)
        plot!(plt, x, hcat(ys...);
            label="",
            linecolor=:blue,
            linewidth=0.5,
            markersize=1,
            marker=:x,
            markercolor=:blue,
        )
        savefig(plt, joinpath(plot_dir, "density_samples_lines_$d.pdf"))
        x_prev = x
    end
end



#
# Simple Conditioning Example
#

let

    local_plot_dir = joinpath(plot_dir, "simple_conditioning")

    rng = MersenneTwister(123456);

    # Generate inputs and pick some to condition on.
    N, S, D = 11, 5, 5
    x = collect(range(0, N-1; length=250))
    x_tr = [1.5, 1.8, 4.7]

    # Generate observations and condition on them.
    y_tr, y_truth = rand([f(x_tr, 1e-9), f(x, 1e-9)])
    f′ = f | (f(x_tr, 1e-9) ← y_tr)

    # Generate samples from the posterior distribution.
    y′ = rand(f′(x, 1e-9), S)
    posterior_marginals = marginals(f′(x, 1e-9))

    plt = plot(ylim=[-4.0, 4.0])

    # Plot sample from the prior.
    plot!(plt, x, y_truth;
        linewidth=1.0,
        linestyle=:dash,
        linecolor=:blue,
        label="",
    )
    savefig(plt, joinpath(local_plot_dir, "sample_from_prior.pdf"))

    # Plot points that we'll observe.
    plot!(plt, x_tr, y_tr;
        linewidth=0,
        marker=:o,
        markercolor=:blue,
        label="",
        markersize=10,
        markerstrokewidth=0,
    )
    savefig(plt, joinpath(local_plot_dir, "obs_points.pdf"))

    plot!(plt, x, y′;
        linecolor=:blue,
        linewidth=0.5,
        label="",
    )
    savefig(plt, joinpath(local_plot_dir, "posterior_samples.pdf"))

    m, σ = mean.(posterior_marginals), std.(posterior_marginals)
    plot!(plt, x, m;
        linewidth=2.0,
        linecolor=:blue,
        label="",
    )
    plot!(plt, x, [m m];
        linewidth=0,
        fillrange=[m .- 3 .* σ, m .+ 3 .* σ],
        fillalpha=0.2,
        fillcolor=:blue,
        label="",
    )
    savefig(plt, joinpath(local_plot_dir, "posterior_marginals.pdf"))
end




















#
# Additive example
#

using Stheno: @model

@model function model()
    f₁ = GP(4.0, eq())
    f₂ = GP(8.0, eq())
    f₃ = f₁ + f₂
    return f₁, f₂, f₃
end




let
    local_plot_dir = joinpath(plot_dir, "additive_decomposition")

    # Randomly sample `N₁` locations at which to measure `f` using `y1`, and `N2` locations
    # at which to measure `f` using `y2`.
    rng, N₁, N₃ = MersenneTwister(123546), 10, 11;
    X₁, X₃ = rand(rng, N₁) * 10, rand(rng, N₃) * 10;
    f₁, f₂, f₃ = model();

    # Define some plotting stuff.
    Np, S = 500, 15;
    Xp = range(-2.5, stop=12.5, length=Np);

    # Generate some toy observations of `f₁` and `f₃`.
    ŷ₁, ŷ₃, y₁_exact, y₂_exact, y₃_exact =
        rand(rng, [f₁(X₁, 1e-9), f₃(X₃, 1e-9), f₁(Xp, 1e-9), f₂(Xp, 1e-9), f₃(Xp, 1e-9)]);

    # Compute the posterior processes.
    (f₁′, f₂′, f₃′) = (f₁, f₂, f₃) | (f₁(X₁)←ŷ₁, f₃(X₃)←ŷ₃);

    # Sample jointly from the posterior over each process.
    f₁′Xp, f₂′Xp, f₃′Xp = rand(rng, [f₁′(Xp, 1e-9), f₂′(Xp, 1e-9), f₃′(Xp, 1e-9)], S);

    # Compute posterior marginals.
    ms1 = marginals(f₁′(Xp));
    ms2 = marginals(f₂′(Xp));
    ms3 = marginals(f₃′(Xp));

    μf₁′, σf₁′ = mean.(ms1), std.(ms1);
    μf₂′, σf₂′ = mean.(ms2), std.(ms2);
    μf₃′, σf₃′ = mean.(ms3), std.(ms3);


    marker_size = 8

    posterior_plot = plot(ylim=[1, 17]);

    #
    # Plot initial samples
    #
    plot!(posterior_plot, Xp, y₁_exact;
        linewidth=1.0,
        linestyle=:dash,
        linecolor=:red,
        label="",
    )
    plot!(posterior_plot, Xp, y₂_exact;
        linewidth=1.0,
        linestyle=:dash,
        linecolor=:green,
        label="",
    )
    plot!(posterior_plot, Xp, y₃_exact;
        linewidth=1.0,
        linestyle=:dash,
        linecolor=:blue,
        label="",
    )
    savefig(posterior_plot, joinpath(local_plot_dir, "sample_paths.pdf"))

    #
    # Plot observations
    #
    scatter!(posterior_plot, X₃, ŷ₃;
        markercolor=:blue,
        markershape=:circle,
        markerstrokewidth=0.0,
        markersize=marker_size,
        markeralpha=0.7,
        label="",
    )
    scatter!(posterior_plot, X₁, ŷ₁;
        markercolor=:red,
        markershape=:circle,
        markerstrokewidth=0.0,
        markersize=marker_size,
        markeralpha=0.7,
        label="",
    )
    savefig(posterior_plot, joinpath(local_plot_dir, "sample_paths_points.pdf"))

    #
    # Plot posterior of third process
    #
    plot!(posterior_plot, Xp, [μf₃′ μf₃′];
        linewidth=0.0,
        fillrange=[μf₃′ .- 3 .* σf₃′, μf₃′ .+ 3 * σf₃′],
        fillalpha=0.3,
        fillcolor=:blue,
        label="",
    )
    plot!(posterior_plot, Xp, f₃′Xp,
        linecolor=:blue,
        linealpha=0.2,
        label="",
    )
    plot!(posterior_plot, Xp, μf₃′;
        linecolor=:blue,
        linewidth=2.0,  
        label="",
    )
    savefig(posterior_plot, joinpath(local_plot_dir, "posterior_3.pdf"))

    #
    # Plot posterior of first process
    #
    plot!(posterior_plot, Xp, [μf₁′ μf₁′];
        linewidth=0.0,
        fillrange=[μf₁′ .- 3 .* σf₁′, μf₁′ .+ 3 * σf₁′],
        fillalpha=0.3,
        fillcolor=:red,
        label="",
    )
    plot!(posterior_plot, Xp, f₁′Xp,
        linecolor=:red,
        linealpha=0.2,
        label="",
    )
    plot!(posterior_plot, Xp, μf₁′;
        linecolor=:red,
        linewidth=2.0,
        label="",
    )
    savefig(posterior_plot, joinpath(local_plot_dir, "posterior_1_3.pdf"))

    #
    # Plot posterior of second process
    #
    plot!(posterior_plot, Xp, [μf₂′ μf₂′];
        linewidth=0.0,
        fillrange=[μf₂′ .- 3 .* σf₂′, μf₂′ .+ 3 * σf₂′],
        fillalpha=0.3,
        fillcolor=:green,
        label="",
    )
    plot!(posterior_plot, Xp, f₂′Xp,
        linecolor=:green,
        linealpha=0.2,
        label="",
    )
    plot!(posterior_plot, Xp, μf₂′;
        linecolor=:green,
        linewidth=2.0,
        label="",
    )
    savefig(posterior_plot, joinpath(local_plot_dir, "additive_decomposition.pdf"))
end


#
# Example: composition
#

@model function composition_model(l)
    f = GP(eq())
    g = stretch(f, l)
    return f, g
end

# How do best visualise this stuff? Do we look at mulitple length scales and produce a gif?
let
    local_plot_dir = joinpath(plot_dir, "composition")

    # Produce GPs and input locations.
    rng, N, Np, S = MersenneTwister(123456), 5, 250, 15
    f, g = composition_model(0.5)
    x, xp = randn(rng, N), range(-5.0, 5.0; length=Np)

    # Generate samples.
    y, y_truth = rand(rng, [g(x, 1e-9), g(xp, 1e-9)])

    # Generate the posterior process.
    g′ = g | (g(x, 1e-9) ← y)

    # Generate posterior samples.
    g′xp = rand(rng, g′(xp, 1e-9), S)

    # Compute posterior marginals.
    msg = marginals(g′(xp))
    m, σ = mean.(msg), std.(msg)

    plt = plot(ylim=[-2.5, 3.2])

    # Plot initial sample.
    plot!(plt, xp, y_truth;
        linewidth=1.0,
        linestyle=:dash,
        linecolor=:green,
        label="",
    )
    savefig(plt, joinpath(local_plot_dir, "sample_path.pdf"))

    # Plot points that we'll observe.
    plot!(plt, x, y;
        linewidth=0,
        marker=:o,
        markercolor=:green,
        label="",
        markersize=10,
        markerstrokewidth=0,
    )
    savefig(plt, joinpath(local_plot_dir, "obs_points.pdf"))

    plot!(plt, xp, g′xp;
        linecolor=:green,
        linewidth=0.5,
        label="",
    )
    savefig(plt, joinpath(local_plot_dir, "posterior_samples.pdf"))

    plot!(plt, xp, m;
        linewidth=2.0,
        linecolor=:green,
        label="",
    )
    plot!(plt, xp, [m m];
        linewidth=0,
        fillrange=[m .- 3 .* σ, m .+ 3 .* σ],
        fillalpha=0.2,
        fillcolor=:green,
        label="",
    )
    savefig(plt, joinpath(local_plot_dir, "posterior_marginals.pdf"))

end


# Distinction between true thing, what I've observed, and what I know about the true thing given what I've observed
# Make it explicit that we're conditioning jointly on _all_ observations at the same time
# ADD SOME MOTIVATION IN A SLIDE!

# # Sample model 1 for slide.
# @model function model()
#     f₁ = GP(m₁, c₁)
#     f₂ = GP(m₂, c₂)
#     f₃ = f₁ + f₂
#     f₄ = f₁ | (f₃(x) ← y)
# end


# chandrasekhar recursions in kalman filtering


# ethanmatlin@gmail.com
# federal reserve guy - SMC package. Collaborate with Turing?




















