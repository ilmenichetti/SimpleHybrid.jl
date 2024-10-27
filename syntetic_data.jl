
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
