using Random, Stheno, Flux, Plots, Zygote, ProgressMeter
plotly();



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
θ_cu = Dict([(key, cu(val)) for (key, val) in θ])

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
# GP + Sawtooth Model
#

θ = Dict(
    :W1 => 0.1 .* randn(Dh, N),
    :b1 => 0.1 .* randn(Dh),
    :W2 => 0.1 .* randn(Dh, Dh),
    :b2 => 0.1 .* randn(Dh),
    :W3 => 0.1 .* randn(1, Dh),
    :b3 => 0.1 .* randn(1),
    :log_σw => 0.1 .* randn(1),
    :log_σb => 0.1 .* randn(1),
    :log_lw => 0.1 .* randn(1),
    :log_lb => 0.1 .* randn(1),
)

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

    σw, σb = exp(first(θ[:log_σw])) + 0.01, exp(first(θ[:log_σb])) + 0.01
    lw, lb = exp(first(θ[:log_lw])) + 0.01, exp(first(θ[:log_lb])) + 0.01

    w = σw * stretch(GP(eq()), lw)
    b = σb * stretch(GP(eq()), lb)
    f = b + w * DataTransform(predict(θ, Y))
    return w, b, f
end

#
# Generate some toy data
#








#
# Learn optimal parameters
#

function nlml_gp(θ, Y, t, y)
    w, b, f = modulation_gps(θ, Y)
    return -logpdf(f(t), y)
end

t = collect(1:size(Y, 2))
nlml_gp(θ, Y, t, y_target)


eta = 1e-3;
opt = Dict([(key, ADAM(eta, (0.9, 0.999))) for key in keys(θ)])

iters = 10;
p = ProgressMeter.Progress(iters);

for iter in 1:iters
    ls, back = Zygote.forward(nlml_gp, θ, Y, t, y_target)
    dθ, _, _ = back(1.0)
    for key in keys(opt)
        Flux.Optimise.update!(opt[key], θ[key], dθ[key])
    end
    ProgressMeter.next!(p; showvalues=[(:iter, iter), (:lml, -ls)])
end


#
# Make predictions
#
