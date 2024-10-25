using Flux

# Define helper function to create RUE and Rb models
function Dense_RUE_Rb(in_dim; neurons=15, out_dim=1, affine=true)
    return Flux.Chain(
        BatchNorm(in_dim, affine=affine),
        Dense(in_dim => neurons, relu),
        Dense(neurons => out_dim, σ)
    )
end

using DataFrames, AxisKeys, Chain, Random
using Chain: @chain
using DataFramesMeta

# Create synthetic data with symbolic row keys
function gen_dk(; seed=123)
    Random.seed!(seed)
    
    # Generate random data and add additional columns based on transformations
    df = DataFrame(rand(Float32, 1000, 5), :auto)
    df = @chain df begin
        rename!(_, [:x1, :x2, :x3, :x4, :x5])  # Rename columns for easy reference
        @transform :a_syn = exp.(-5.0f0 .* (:x2 .- 0.7f0).^2.0f0) .+ :x3 ./ 10.0f0
        @aside b = 2.0f0
        @transform :obs = :a_syn .* :x1 .+ b
        @transform :pred_syn = :obs
        @transform :seqID = Float32.(repeat(1:100, inner=10))  # Convert seqID to Float32
    end

    # Define row and column keys
    row_keys = [:TA, :VPD, :SW_POT_sm_diff, :SW_POT_sm, :SW_IN, :a_syn, :obs, :pred_syn, :seqID]
    col_keys = 1:size(df, 1)
    
    # Convert to a KeyedArray for easier manipulation
    data_matrix = Matrix{Float32}(df)'
    dk = KeyedArray(data_matrix, row=row_keys, col=col_keys)
    
    return dk
end

# Define model struct
struct FluxPartModel_Q10
    RUE_chain::Flux.Chain
    Rb_chain::Flux.Chain
    Q10::Vector{Float32}
end

# Basic constructor for `FluxPartModel_Q10`
function FluxPartModel_Q10(; Q10=[1.5f0], neurons=15)
    RUE_chain = Dense_RUE_Rb(2; neurons)   # Assume 2 predictors for simplicity
    Rb_chain = Dense_RUE_Rb(2; neurons)    # Assume 2 predictors for simplicity
    return FluxPartModel_Q10(RUE_chain, Rb_chain, Q10)
end

# Model inference method
function (m::FluxPartModel_Q10)(dk)
    # Access inputs by fixed index positions
    RUE_input = dk[[1, 2], :]   # Assuming `:TA` is at index 1 and `:VPD` at index 2
    Rb_input = dk[[3, 4], :]    # Assuming `:SW_POT_sm_diff` is at index 3 and `:SW_POT_sm` at index 4

    # Print shapes before transposing
    #println("RUE_input shape before transpose: ", size(RUE_input))
    #println("Rb_input shape before transpose: ", size(Rb_input))

    # Pass inputs through the model directly
    RUE = m.RUE_chain(RUE_input)[:, 1]  #RUE (Radiative Use Efficiency)
    Rb = 100.0f0 * m.Rb_chain(Rb_input)[:, 1]  #Rb (Respiration or Basal Respiration)

    # Access forcing variables by their fixed indices
    sw_in = dk[5, :]  # Assuming `:SW_IN` is at index 5
    ta = dk[1, :]     # Assuming `:TA` is at index 1

    # Calculate GPP and Reco
    GPP = sw_in .* RUE ./ 12.011f0
    Reco = Rb .* m.Q10[1] .^ (0.1f0 * (ta .- 15.0f0))

    return (; NEE = Reco - GPP)
end

# Allow the model to work with Flux's `train!` function
Flux.@functor FluxPartModel_Q10
