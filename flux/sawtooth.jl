using Random, Stheno, Flux, Plots, Zygote, ProgressMeter, FDM, Optim, StatsFuns
plotly();


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
sawtooth(x) = mod(x, 1)

# Plot it for sanity.
x = collect(range(1.0, 10.0; length=1000));

# reformat `x` into an `N x length(x)` matrix
x = collect(range(1.0, 10.0; length=1000));
y = sawtooth.(x);

# Convert to autoregressive-format.
N = 9;
y_target, y_pad = y[N+1:end], y[1:N];
Y = to_ar_data(y_target, N, y_pad);

function predict(θ, Y)
    return vec(Chain(
        x->Dense(θ[:W1], θ[:b1])(x),
        x->Dense(θ[:W2], θ[:b2])(x) + x,
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

loss(θ, Y, y) = mean(abs2, y - predict(θ, Y))


#
# Train NN
#

eta = 1e-3;
opt = Dict([(key, ADAM(eta, (0.9, 0.999))) for key in keys(θ)])

iters = 1_000;
p = ProgressMeter.Progress(iters);

for iter in 1:iters
    ls, back = Zygote.forward(loss, θ, Y, y_target)
    dθ, _, _ = back(1.0)
    for key in keys(opt)
        Flux.Optimise.update!(opt[key], θ[key], dθ[key])
    end
    ProgressMeter.next!(p; showvalues=[(:iter, iter), (:sqrt_loss, sqrt(ls))])
end



#
# Predict using NN
#

ypr = predict(θ, Y);
plot(x[N+1:end], y_target; label="Truth", linecolor=:red);
plot!(x[N+1:end], ypr; label="Pr", linecolor=:blue)


plot(x[N+1:end], y_target - ypr)







#
# GP-modulated sawtooth model
#

θ_gp = merge(
    θ, 
    Dict(
        :log_σw => 0.1 .* randn(1),
        :log_σb => 0.1 .* randn(1),
        :log_lw => 0.1 .* randn(1),
        :log_lb => 0.1 .* randn(1),
    ),
);

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

    w = σw * stretch(GP(eq()), lw) + 1
    b = σb * stretch(GP(eq()), lb)
    f = b + w * DataTransform(predict(θ, Y))
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
    ),
);

T = 1000;
t = collect(1:size(Y, 2));
w, b, f = modulation_gps(θ0, Y);
ws, bs, ys = rand([w(t, 1e-6), b(t, 1e-6), f(t, 1e-3)]);

plot(ys; linecolor=:red, label="y");
plot!(ws; linecolor=:blue, label="w");
plot!(bs; linecolor=:green, label="b")

Ntr = 500;
Ytr, Yte = Y[:, 1:Ntr], Y[:, Ntr+1:end];
ystr, yste = ys[1:Ntr], ys[Ntr+1:end];
ttr, tte = t[1:Ntr], t[Ntr+1:end];

#
# Learn optimal parameters
#

function nlml_gp(θ, Y, t, y)
    w, b, f = modulation_gps(θ, Y)
    return -logpdf(f(t, 1e-3), y)
end

eta = 1e-3;
opt = Dict([(key, ADAM(eta, (0.9, 0.999))) for key in keys(θ)]);

iters = 10;
p = ProgressMeter.Progress(iters);

for iter in 1:iters
    ls, back = Zygote.forward(nlml_gp, θ0, Ytr, ttr, ystr)
    dθ, _, _ = back(1.0)
    for key in keys(opt)
        Flux.Optimise.update!(opt[key], θ[key], dθ[key])
    end
    ProgressMeter.next!(p; showvalues=[(:iter, iter), (:nlml, ls)])
end


#
# Make predictions
#

w, b, f = modulation_gps(θ0, Y);
w′, b′, f′ = (w, b, f) | (f(ttr, 1e-3) ← ystr);

w′s, b′s, f′s = rand([w′(t, 1e-9), b′(t, 1e-9), f′(t, 1e-9)], 11);

ms_w′ = marginals(w′(t, 1e-9));
ms_b′ = marginals(b′(t, 1e-9));
ms_f′ = marginals(f′(t, 1e-9));

plotly();
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
plot!(posterior_plot, t, mean.(ms_f′);
    linecolor=:red,
    linewidth=2.0,
    label="f");
plot!(posterior_plot, t, mean.(ms_b′);
    linecolor=:green,
    linewidth=2.0,
    label="b");
plot!(posterior_plot, t, mean.(ms_w′);
    linecolor=:blue,
    linewidth=2.0,
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

display(posterior_plot);






#
# Learning the NN as well.
#

θ_1 = Dict(
    :W1 => 0.1 .* randn(Dh, N),
    :b1 => 0.1 .* randn(Dh),
    :W2 => 0.1 .* randn(Dh, Dh),
    :b2 => 0.1 .* randn(Dh),
    :W3 => 0.1 .* randn(1, Dh),
    :b3 => 0.1 .* randn(1),
    :log_σw => [log(0.1)],
    :log_σb => [log(0.1)],
    :log_lw => [log(1e-2)],
    :log_lb => [log(1e-2)],
);


eta = 1e-3;
opt = Dict([(key, ADAM(eta, (0.9, 0.999))) for key in keys(θ_1)]);

iters = 500;
p = ProgressMeter.Progress(iters);

for iter in 1:iters
    ls, back = Zygote.forward(nlml_gp, θ_1, Ytr, ttr, ystr)
    dθ, _, _ = back(1.0)
    for key in keys(opt)
        Flux.Optimise.update!(opt[key], θ_1[key], dθ[key])
    end
    ProgressMeter.next!(p; showvalues=[(:iter, iter), (:nlml, ls)])
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
    ms_f′ = marginals(f′(t, 1e-9));

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
    plot!(posterior_plot, t, predict(θ_1, Y);
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
end








#
# Fit approximately-periodic GP to the data
#

@model function gp_model(θ)

    l_per, period = to_pos(first(θ[:log_l_per])), to_pos(first(θ[:log_period]))
    l_mod, σ_per = to_pos(first(θ[:log_l_mod])), to_pos(first(θ[:log_σ_per]))
    f_per = σ_per * GP(periodic(eq(α=l_per), 1 / period) * eq(α=l_mod))

    l_b, σ_b = to_pos(first(θ[:log_l_b])), to_pos(first(θ[:log_σ_b]))
    b = σ_b * GP(eq(α=l_b))

    f = b + f_per
    return b, f_per, f
end


# θ_gp = Dict(
#     :log_l_per => [logexpm1(1.0)],
#     :log_period => [logexpm1(100)],
#     :log_l_mod => [logexpm1(1e-2)],
#     :log_σ_per => [logexpm1(1)],
#     :log_l_b => [logexpm1(1e-2)],
#     :log_σ_b => [logexpm1(1e-1)],
# );

θ_gp = Dict(
    :log_l_per => [log(1.0)],
    :log_period => [log(100)],
    :log_l_mod => [log(1e-2)],
    :log_σ_per => [log(1)],
    :log_l_b => [log(1e-2)],
    :log_σ_b => [log(1e-1)],
);

function nlml_gp_model(θ, t, y)
    _, _, f = gp_model(θ)
    return -logpdf(f(t, 1e-3), y)
end


# eta = 1e-3;
# opt = Dict([(key, ADAM(eta, (0.9, 0.999))) for key in keys(θ_gp)]);
# let
#     iters = 100;
#     p = ProgressMeter.Progress(iters);

#     for iter in 1:iters
#         ls, back = Zygote.forward(nlml_gp_model, θ_gp, ttr, ystr)
#         dθ, _, _ = back(1.0)
#         for key in keys(opt)
#             Flux.Optimise.update!(opt[key], θ_gp[key], dθ[key])
#         end
#         ProgressMeter.next!(p; showvalues=[(:iter, iter), (:nlml, ls)])
#     end
# end

# Construct Optim objective and gradient evaluation.
using FDM: to_vec
vec_θ, back_θ = FDM.to_vec(θ_gp)

nlml_gp_model_optim(vec_θ) = nlml_gp_model(back_θ(vec_θ), ttr, ystr)
function dθ_nlml_gp_model_optim(vec_θ)
    θ = back_θ(vec_θ)
    dθ = first(Zygote.gradient(θ->nlml_gp_model(θ, ttr, ystr), θ))
    return first(to_vec(dθ))
end

options = Optim.Options(show_trace = true, iterations = 10);
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
    plot!(posterior_plot, t, [mean.(ms_f_per′) mean.(ms_f_per′)];
        linewidth=0.0,
        fillrange=[mean.(ms_f_per′) .- 3 .* std.(ms_f_per′), mean.(ms_f_per′) .+ 3 * std.(ms_f_per′)],
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
    plot!(posterior_plot, t, f_per′s,
        linecolor=:blue,
        linealpha=0.2,
        label="");

    # Plot posterior means
    plot!(posterior_plot, t, mean.(ms_f′);
        linecolor=:red,
        linewidth=1,
        label="f");
    plot!(posterior_plot, t, mean.(ms_b′);
        linecolor=:green,
        linewidth=1,
        label="b");
    plot!(posterior_plot, t, mean.(ms_f_per′);
        linecolor=:blue,
        linewidth=1,
        label="f_per");

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
    savefig(posterior_plot, "flux/gp-only-sawtooth-learning.pdf")
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
