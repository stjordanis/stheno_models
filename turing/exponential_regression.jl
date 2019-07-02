using Turing, Stheno, Plots, Flux
using Stheno: TuringGPC
using Turing: @model


#
# Define a vaguely interesting non-Gaussian regression problem
#

N = 100;
x = collect(range(-5.0, 5.0; length=N));

@model exponential_regression(y) = begin
    y = Vector{Real}(undef, N)
    f = GP(eq(), TuringGPC())
    fx ~ f(x, 1e-6)
    ϕx = softplus.(fx)
    for n in 1:N
        y[n] ~ Exponential(ϕx[n])
    end
    return fx, y
end


#
# Sample from the prior and plot it
#

fx, y = exponential_regression()();

plotly();
plot(x, fx);
plot!(x, y)


#
# Do posterior inference via HMC (NUTS)
#






