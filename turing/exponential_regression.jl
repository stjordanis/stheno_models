using Turing, Stheno, Plots, Flux
using Stheno: TuringGPC
using Turing: @model

Turing.setadbackend(:reverse_diff)



#
# Define a vaguely interesting non-Gaussian regression problem
#

N = 25;
x = collect(range(-5.0, 5.0; length=N));
x_pr = collect(range(-5.0, 5.0; length=100));

@model gaussian_regression(y) = begin
    f = GP(eq(), TuringGPC())
    fx ~ f(x, 1e-9)
    fx_pr ~ f(x_pr, 1e-9)
    y ~ Product(Normal.(fx, 0.1))
    return fx, fx_pr, y
end



#
# Sample from the prior and plot it
#

fx, fx_pr, y = gaussian_regression()();

plotly();
plot(x, fx; label="Latent at observations");
scatter!(x, y; label="Observations");
plot!(x_pr, fx_pr; label="")



#
# Do posterior inference via HMC (NUTS)
#

chain = sample(gaussian_regression(y), NUTS(2, 0, 0.6));
fx_slices = get(chain, :fx)[1];
fx_samples = Float64.(hcat(fx_slices...));

plot(x, fx_samples[200:50:end, :]'; label="", linecolor=:red);
scatter!(x, y);
plot!(x, fx; linecolor=:blue)



#
# Logistic regression equivalent example
#

x = collect(range(-5.0, 5.0; length=100));

@model logistic_regression(y) = begin
    f = 5.0 * GP(eq(), TuringGPC())
    fx ~ f(x, 1e-9)
    y ~ Product(Bernoulli.(σ.(fx)))
    return fx, y
end

fx, y = logistic_regression()();

chain = sample(logistic_regression(y), NUTS(1_000, 200, 0.6));
fx_slices = get(chain, :fx)[1];
fx_samples = Float64.(hcat(fx_slices...));

plot(x, fx_samples[200:50:end, :]'; label="", linecolor=:red);
scatter!(x, y);
plot!(x, fx; linecolor=:blue)


#
# Poisson regression
#

x = vcat(range(-5.0, -2.5; length=13), range(1.0, 2.0; length=12));
xpr = collect(range(-5.0, 5.0; length=50));


@model poisson_regression(y) = begin
    f = GP(eq(), TuringGPC())
    fx ~ f(x, 1e-9)
    fx_pr ~ f(xpr, 1e-9)
    y ~ Product(Poisson.(exp.(fx)))
    return fx, fx_pr, y
end

fx, fx_pr, y = poisson_regression()();

plot(x, y);
plot!(x, fx);
plot!(xpr, fx_pr)

chain = sample(poisson_regression(y), NUTS(10, 0, 0.6));
fx_slices = get(chain, :fx)[1];
fx_samples = Float64.(hcat(fx_slices...));

plot(x, fx_samples[:, :]'; label="", linecolor=:red);
scatter!(x, y);
plot!(x, fx; linecolor=:blue)



