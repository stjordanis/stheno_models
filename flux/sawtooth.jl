using Random, Stheno, Flux, Plots, Zygote, ProgressMeter, FDM, Optim, StatsFuns
pyplot();


to_pos(θ::Real) = exp(θ) + 1e-9

#
# Util function
#

function to_ar_data(x::Vector{T}, N::Int, pad::Vector{T}) where {T}
    @assert length(pad) == N
    X = Matrix{T}(undef, N, length(x))
    x′ = vcat(pad, x)
    for p in 1:length(x)
        for n in 1:N
            X[n, p] = x′[N + p - n]
        end
    end
    return reverse(X; dims=1)
end

#
# Set up sawtooth.
#

# Define sawtooth with unit period.
sawtooth(x) = mod(x, 1) - 0.5

# Generate data to plot
x = collect(1.0:0.01:10.0);
# x = collect(range(1.0, 10.0; length=1000));
y = sawtooth.(x);

plt = plot();
plot!(plt, x, y; linecolor=:blue, linewidth=2, label="");
savefig(plt, "flux/vanilla_sawtooth.pdf");

# Convert to autoregressive-format.
N = 5;
y_target, y_pad = y[N+1:end], y[1:N];
Y = to_ar_data(y_target, N, y_pad);

# Zygote.refresh();
function predict(θ, Y, nonlinearity)
    return vec(Chain(
        x->Dense(θ[:W1], θ[:b1], nonlinearity)(x),
        x->Dense(θ[:W2], θ[:b2], nonlinearity)(x) + x,
        x->Dense(θ[:W3], θ[:b3])(x),
    )(Y))
end

Dh = 50
θ = Dict(
    :W1 => 0.1 .* randn(Dh, N),
    :b1 => 0.1 .* randn(Dh),
    :W2 => 0.1 .* randn(Dh, Dh),
    :b2 => 0.1 .* randn(Dh),
    :W3 => 0.1 .* randn(1, Dh),
    :b3 => 0.1 .* randn(1),
)

loss(θ, Y, y, nonlinearity) = mean(abs2, y - predict(θ, Y, nonlinearity))


#
# Train NN
#

eta = 1e-3;
opt = Dict([(key, ADAM(eta, (0.9, 0.999))) for key in keys(θ)])

iters = 10_000;
p = ProgressMeter.Progress(iters);

for iter in 1:iters
    ls, back = Zygote.forward(loss, θ, Y, y_target, relu)
    dθ, _, _ = back(1.0)
    for key in keys(opt)
        Flux.Optimise.update!(opt[key], θ[key], dθ[key])
    end
    ProgressMeter.next!(p; showvalues=[(:iter, iter), (:sqrt_loss, sqrt(ls))])
end



#
# Predict using NN
#

ypr = predict(θ, Y, relu);

plot(x[N+1:end], y_target; label="Truth", linecolor=:red);
plot!(x[N+1:end], ypr; label="Pr", linecolor=:blue)

plot(x[N+1:end], y_target - ypr)







#
# GP-modulated sawtooth model
#

using Stheno: @model

struct DataTransform{TD<:Vector}
    D::TD
end
Base.Broadcast.broadcasted(dt::DataTransform, x::Vector{Int}) = dt.D[x]
Base.map(dt::DataTransform, x::Vector{Int}) = dt.(x)
Zygote.@adjoint function map(dt::DataTransform, x::Vector{Int})
    y, back = Zygote.forward(getindex, dt.D, x)
end
Zygote.@adjoint (::Type{T})(dt::Vector) where T<:DataTransform = T(dt), ȳ -> (ȳ,)

@model function modulation_gps(θ, Y)

    σw, σb = exp(first(θ[:log_σw])) + 1e-6, exp(first(θ[:log_σb])) + 1e-6
    lw, lb = exp(first(θ[:log_lw])) + 1e-6, exp(first(θ[:log_lb])) + 1e-6

    w = σw * stretch(GP(first(θ[:m_w]), eq()), lw) + 1
    b = σb * stretch(GP(first(θ[:m_b]), eq()), lb)
    f = b + w * DataTransform(predict(θ, Y, relu))
    return w, b, f
end



#
# Generate some toy data
#

θ0 = merge(
    θ,
    Dict(
        :log_σw => [log(0.1)],
        :log_σb => [log(0.1)],
        :log_lw => [log(1e-2)],
        :log_lb => [log(1e-2)],
        :m_w => [0.0],
        :m_b => [0.0],
    ),
);

T = 1000;
t = collect(1:size(Y, 2));
w, b, _ = modulation_gps(θ0, Y);
ws, bs = rand(MersenneTwister(123456), [w(t, 1e-6), b(t, 1e-6)]);
fs = ws .* sawtooth.(x[N+1:end]) .+ bs;
ys = fs .+ sqrt(1e-3) .* randn(MersenneTwister(123456), length(fs));


plot(ys; linecolor=:red, label="y");
plot!(ws; linecolor=:blue, label="w");
plot!(bs; linecolor=:green, label="b")

Ntr = 500;
Ytr, Yte = Y[:, 1:Ntr], Y[:, Ntr+1:end];
ystr, yste = ys[1:Ntr], ys[Ntr+1:end];
ttr, tte = t[1:Ntr], t[Ntr+1:end];



#
# Learning the NN as well.
#

Zygote.refresh()
function nlml_gp(θ, Y, t, y)

    reg_loss = 0.0
    for key in keys(θ)
        reg_loss += 1e-2 * sum(abs2, θ[key])
    end

    w, b, f = modulation_gps(θ, Y)
    return -logpdf(f(t, exp(first(θ[:log_σ_noise]))), y)
end

rng = MersenneTwister(123457);
θ_1 = Dict(
    :W1 => 0.01 .* randn(rng, Dh, N),
    :b1 => 0.01 .* randn(rng, Dh),
    :W2 => 0.01 .* randn(rng, Dh, Dh),
    :b2 => 0.01 .* randn(rng, Dh),
    :W3 => 0.01 .* randn(rng, 1, Dh),
    :b3 => 0.01 .* randn(rng, 1),
    :log_σw => [log(0.1)],
    :log_σb => [log(0.1)],
    :log_lw => [log(1e-2)],
    :log_lb => [log(1e-2)],
    :log_σ_noise => [log(1e-3)],
    :m_w => [1e-6],
    :m_b => [1e-6],
);


eta = 1e-3;
opt = Dict([(key, ADAM(eta, (0.9, 0.999))) for key in keys(θ_1)]);

# Run this block a few times to actually optimise. Look at the learning curve (plotted
# below) to assess convergence.
iters = 10_000;
p = ProgressMeter.Progress(iters);
ls = Vector{Float64}(undef, iters);
for iter in 1:iters
    l, back = Zygote.forward(nlml_gp, θ_1, Ytr, ttr, ystr)
    ls[iter] = l
    dθ, _, _, _ = back(1.0)

    for key in keys(opt)
        Flux.Optimise.update!(opt[key], θ_1[key], dθ[key])
    end

    show_vals = [
        (:iter, iter),
        (:nlml, l),
        (:m_w, first(θ_1[:m_w])),
        (:dθ_m_w, first(dθ[:m_w])),
    ]
    ProgressMeter.next!(p; showvalues=show_vals)
end


#
# Make predictions
#

let

    w, b, f = modulation_gps(θ_1, Y);
    w′, b′, f′ = (w, b, f) | (f(ttr, 1e-3) ← ystr);

    w′s, b′s, f′s = rand([w′(t, 1e-9), b′(t, 1e-9), f′(t, 1e-9)], 11);

    ms_w′ = marginals(w′(t, 1e-9));
    ms_b′ = marginals(b′(t, 1e-9));
    ms_f′ = marginals(f′(t, exp(first(θ_1[:log_σ_noise]))));

    pyplot();
    posterior_plot = plot();

    # Plot posterior marginal variances
    plot!(posterior_plot, t, [mean.(ms_f′) mean.(ms_f′)];
        linewidth=0.0,
        fillrange=[mean.(ms_f′) .- 3 .* std.(ms_f′), mean.(ms_f′) .+ 3 * std.(ms_f′)],
        fillalpha=0.3,
        fillcolor=:red,
        label="");
    plot!(posterior_plot, t, [mean.(ms_b′) mean.(ms_b′)];
        linewidth=0.0,
        fillrange=[mean.(ms_b′) .- 3 .* std.(ms_b′), mean.(ms_b′) .+ 3 * std.(ms_b′)],
        fillalpha=0.3,
        fillcolor=:green,
        label="");
    plot!(posterior_plot, t, [mean.(ms_w′) mean.(ms_w′)];
        linewidth=0.0,
        fillrange=[mean.(ms_w′) .- 3 .* std.(ms_w′), mean.(ms_w′) .+ 3 * std.(ms_w′)],
        fillalpha=0.3,
        fillcolor=:blue,
        label="");

    # Plot joint posterior samples
    plot!(posterior_plot, t, f′s,
        linecolor=:red,
        linealpha=0.2,
        label="");
    plot!(posterior_plot, t, b′s,
        linecolor=:green,
        linealpha=0.2,
        label="");
    plot!(posterior_plot, t, w′s,
        linecolor=:blue,
        linealpha=0.2,
        label="");

    # Plot posterior means
    plot!(posterior_plot, t, predict(θ_1, Y, relu);
        linecolor=:black,
        linewidth=1,
        label="nn");
    plot!(posterior_plot, t, mean.(ms_f′);
        linecolor=:red,
        linewidth=1,
        label="f");
    plot!(posterior_plot, t, mean.(ms_b′);
        linecolor=:green,
        linewidth=1,
        label="b");
    plot!(posterior_plot, t, mean.(ms_w′);
        linecolor=:blue,
        linewidth=1,
        label="w");

    # Plot observations
    scatter!(posterior_plot, ttr, ystr;
        markercolor=:black,
        markershape=:circle,
        markerstrokewidth=0.0,
        markersize=2,
        markeralpha=0.5,
        label="");
    scatter!(posterior_plot, tte, yste;
        markercolor=:purple,
        markershape=:circle,
        markerstrokewidth=0.0,
        markersize=2,
        markeralpha=0.3,
        label="");

    # display(posterior_plot);
    savefig(posterior_plot, "flux/gp-sawtooth-learning.pdf")

    #
    # Plot learning curve
    #
    learning_curve = plot();
    plot!(learning_curve, ls)
    savefig(learning_curve, "flux/learning-curve-gp-sawtooth-learning.pdf")


    #
    # Plot only predictions over the training and test data.
    #
    prediction_plot = plot()
    plot!(prediction_plot, t, [mean.(ms_f′) mean.(ms_f′)];
        linewidth=0.0,
        fillrange=[mean.(ms_f′) .- 3 .* std.(ms_f′), mean.(ms_f′) .+ 3 * std.(ms_f′)],
        fillalpha=0.3,
        fillcolor=:red,
        label="");
    plot!(prediction_plot, t, f′s,
        linecolor=:red,
        linealpha=0.2,
        label="");
    plot!(prediction_plot, t, predict(θ_1, Y, relu);
        linecolor=:black,
        linewidth=1,
        label="nn");
    plot!(prediction_plot, t, mean.(ms_f′);
        linecolor=:red,
        linewidth=1,
        label="f");

    scatter!(prediction_plot, ttr, ystr;
        markercolor=:black,
        markershape=:circle,
        markerstrokewidth=0.0,
        markersize=2,
        markeralpha=0.5,
        label="");
    scatter!(prediction_plot, tte, yste;
        markercolor=:purple,
        markershape=:circle,
        markerstrokewidth=0.0,
        markersize=2,
        markeralpha=0.3,
        label="");
    savefig(prediction_plot, "flux/gp-sawtooth-learning-preds.pdf")
end



#
# Fit periodic GP to the data
#

# @model function gp_model(θ)

#     l_per, period = to_pos(first(θ[:log_l_per])), to_pos(first(θ[:log_period]))
#     l_mod, σ_per = to_pos(first(θ[:log_l_mod])), to_pos(first(θ[:log_σ_per]))
#     f_per = σ_per * GP(periodic(eq(α=l_per), 1 / period) * eq(α=l_mod))

#     l_b, σ_b = to_pos(first(θ[:log_l_b])), to_pos(first(θ[:log_σ_b]))
#     b = σ_b * GP(eq(α=l_b))

#     f = b + f_per
#     return b, f_per, f
# end

@model function gp_model(θ)

    l_per, period = to_pos(first(θ[:log_l_per])), to_pos(first(θ[:log_period])) * 100
    σ_per = to_pos(first(θ[:log_σ_per]))
    f_per = σ_per * GP(periodic(eq(α=l_per), 1 / period))

    l_b, σ_b = to_pos(first(θ[:log_l_b])), to_pos(first(θ[:log_σ_b]))
    b = σ_b * GP(eq(α=l_b))

    f = b + f_per
    return b, f_per, f
end

θ_gp = Dict(
    :log_l_per => [log(1.0)],
    :log_period => [log(0.5)],
    # :log_l_mod => [log(1e-2)],
    :log_σ_per => [log(0.5)],
    :log_l_b => [log(1e-1)],
    :log_σ_b => [log(0.5)],
    :log_σ_obs => [log(1e-1)],
);

function nlml_gp_model(θ, t, y)
    _, _, f = gp_model(θ)
    return -logpdf(f(t, to_pos(first(θ[:log_σ_obs]))), y)
end

# Construct Optim objective and gradient evaluation.
using FDM: to_vec
vec_θ, back_θ = FDM.to_vec(θ_gp)

nlml_gp_model_optim(vec_θ) = nlml_gp_model(back_θ(vec_θ), ttr, ystr)
function dθ_nlml_gp_model_optim(vec_θ)
    θ = back_θ(vec_θ)
    dθ = first(Zygote.gradient(θ->nlml_gp_model(θ, ttr, ystr), θ))
    return first(to_vec(dθ))
end

options = Optim.Options(show_trace = true, iterations = 25);
results = optimize(
    nlml_gp_model_optim,
    dθ_nlml_gp_model_optim,
    vec_θ,
    LBFGS(),
    options;
    inplace = false,
)
θ_gp_opt = back_θ(results.minimizer);

θ_natural = Dict([(key, [exp(first(val))]) for (key, val) in θ_gp_opt]);

#
# Make predictions
#

let

    b, f_per, f = gp_model(θ_gp_opt);
    b′, f_per′, f′ = (b, f_per, f) | (f(ttr, 1e-3) ← ystr);

    b′s, f_per′s, f′s = rand([b′(t, 1e-9), f_per′(t, 1e-9), f′(t, 1e-9)], 11);

    ms_b′ = marginals(b′(t, 1e-9));
    ms_f_per′ = marginals(f_per′(t, 1e-9));
    ms_f′ = marginals(f′(t, 1e-9));

    pyplot();
    posterior_plot = plot(; ylim=(-0.5, 1.5));
    posterior_plot_f = plot(; ylim=(-0.5, 1.5), yaxis=false);

    # Plot posterior marginal variances
    plot!(posterior_plot, t, [mean.(ms_b′) mean.(ms_b′)];
        linewidth=0.0,
        fillrange=[mean.(ms_b′) .- 3 .* std.(ms_b′), mean.(ms_b′) .+ 3 * std.(ms_b′)],
        fillalpha=0.3,
        fillcolor=:green,
        label="");
    plot!(posterior_plot, t, [mean.(ms_f_per′) mean.(ms_f_per′)];
        linewidth=0.0,
        fillrange=[mean.(ms_f_per′) .- 3 .* std.(ms_f_per′), mean.(ms_f_per′) .+ 3 * std.(ms_f_per′)],
        fillalpha=0.3,
        fillcolor=:blue,
        label="");
    plot!(posterior_plot_f, t, [mean.(ms_f′) mean.(ms_f′)];
        linewidth=0.0,
        fillrange=[mean.(ms_f′) .- 3 .* std.(ms_f′), mean.(ms_f′) .+ 3 * std.(ms_f′)],
        fillalpha=0.3,
        fillcolor=:red,
        label="");

    # Plot joint posterior samples
    plot!(posterior_plot, t, b′s,
        linecolor=:green,
        linealpha=0.2,
        label="");
    plot!(posterior_plot, t, f_per′s,
        linecolor=:blue,
        linealpha=0.2,
        label="");
    plot!(posterior_plot_f, t, f′s,
        linecolor=:red,
        linealpha=0.2,
        label="");

    # Plot posterior means
    plot!(posterior_plot, t, mean.(ms_b′);
        linecolor=:green,
        linewidth=1,
        label="b");
    plot!(posterior_plot, t, mean.(ms_f_per′);
        linecolor=:blue,
        linewidth=1,
        label="f_per");
    plot!(posterior_plot_f, t, mean.(ms_f′);
        linecolor=:red,
        linewidth=1,
        label="f");

    # Plot observations
    scatter!(posterior_plot_f, ttr, ystr;
        markercolor=:black,
        markershape=:circle,
        markerstrokewidth=0.0,
        markersize=2,
        markeralpha=0.5,
        label="");
    scatter!(posterior_plot_f, tte, yste;
        markercolor=:purple,
        markershape=:circle,
        markerstrokewidth=0.0,
        markersize=2,
        markeralpha=0.5,
        label="");

    # display(posterior_plot);
    posterior_plot_joint = plot(posterior_plot, posterior_plot_f);
    savefig(posterior_plot_joint, "flux/periodic-gp-only-sawtooth-learning.pdf")
end




#
# Train NN alone on data
#

θ_nn = Dict(
    :W1 => 0.1 .* randn(Dh, N),
    :b1 => 0.1 .* randn(Dh),
    :W2 => 0.1 .* randn(Dh, Dh),
    :b2 => 0.1 .* randn(Dh),
    :W3 => 0.1 .* randn(1, Dh),
    :b3 => 0.1 .* randn(1),
)

eta = 1e-3;
opt = Dict([(key, ADAM(eta, (0.9, 0.999))) for key in keys(θ_nn)])

iters = 1_000;
p = ProgressMeter.Progress(iters);

for iter in 1:iters
    ls, back = Zygote.forward(loss, θ_nn, Ytr, ystr)
    dθ, _, _ = back(1.0)
    for key in keys(opt)
        Flux.Optimise.update!(opt[key], θ_nn[key], dθ[key])
    end
    ProgressMeter.next!(p; showvalues=[(:iter, iter), (:sqrt_loss, sqrt(ls))])
end

let
    pyplot();
    posterior_plot = plot(;
        ylim=(-0.5, 1.5),
    );

    # Plot predictions by NN.
    plot!(posterior_plot, t, predict(θ_nn, Y);
        linecolor=:black,
        linewidth=1,
        label="nn");

    # Plot observations
    scatter!(posterior_plot, ttr, ystr;
        markercolor=:black,
        markershape=:circle,
        markerstrokewidth=0.0,
        markersize=2,
        markeralpha=0.5,
        label="");
    scatter!(posterior_plot, tte, yste;
        markercolor=:purple,
        markershape=:circle,
        markerstrokewidth=0.0,
        markersize=2,
        markeralpha=0.3,
        label="");

    # display(posterior_plot);
    savefig(posterior_plot, "flux/nn-only-sawtooth-learning.pdf")
end
