# test_flux_part_model.jl

using Flux
using Test
using Random
using AxisKeys
using DataFrames

# Include model and data generation functions
include("q10.jl") 

# Generate test data
dk = gen_dk()

# Define the model
model = FluxPartModel_Q10()

# Define target data (e.g., NEE values) and convert to standard arrays
target_data = parent(dk(:obs))  # Extract target variable and remove KeyedArray structure

# Set up data for training as a list of input-target tuples
data = [(parent(dk), target_data)]

# Define the loss function
function loss(x, y)
    predictions = model(x)[:NEE]  # Predict NEE from model
    return sum((predictions .- y).^2) / length(y)  # Mean squared error
end

# Set up the optimizer
opt = Flux.Adam(0.01)

# Training loop
epochs = 100
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), data, opt)
    println("Epoch $epoch complete. Loss: ", loss(parent(dk), target_data))
end

@testset "Model Training" begin
    @test loss(parent(dk), target_data) < 0.1  # Example test condition
end


println("Final Q10 values: ", model.Q10)
