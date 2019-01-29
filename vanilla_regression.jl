using Stheno, Plots, Random
using Stheno: @model

@model model(α, β) = GP(eq(α, β, l))
