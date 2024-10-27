# test_flux_part_model.jl

using Flux
using Test
using Random
using AxisKeys
using DataFrames

# Include model and data generation functions
include("q10.jl") 
include("syntetic_data.jl") 

# Generate test data
    # Structure of `dk` matrix (KeyedArray). A KeyedArray is a data structure from the AxisKeys.jl package, similar 
    # to a standard multi-dimensional array (e.g., a matrix) but has the added benefit of associating keys (labels) with its 
    # rows, columns, or other dimensions. 
    # Each row label corresponds to a specific predictor or derived variable, and each column represents an observation:
    # Row Label         Description
    # :TA               Random data, representing temperature (TA)
    # :VPD              Random data, representing vapor pressure deficit (VPD)
    # :SW_POT_sm_diff   Random data, representing soil potential difference
    # :SW_POT_sm        Random data, representing soil moisture
    # :SW_IN            Random data, representing shortwave incoming radiation (SW_IN)
    # :a_syn            Non-linear function of :x2 and :x3, a synthetic variable
    # :obs              Calculated as a_syn * x1 + b, representing observed values
    # :pred_syn         Synthetic prediction, set equal to obs
    # :seqID            Sequence identifier (1 to 100, repeated 10 times)
dk = gen_dk()

# Define the model
model = FluxPartModel_Q10()

# Define target data (e.g., NEE values) and convert to standard arrays
target_data = parent(dk(:obs))  # Extract target variable and remove KeyedArray structure. target_data needs to be specified as a standard array 
# (or similar structure like a vector or matrix) to meet Flux's syntax requirements for the training function train!.

# Set up data for training as a list of input-target tuples
data = [(parent(dk), target_data)] # parent(dk) flattens the dk array into a standard matrix format, representing all input features. The array is alreaduy 2 d, but
# it is a KeyedArray and this transformation preserves compatibility with what follows

# Define the loss function
# In this case it calculates the Mean Squared Error (MSE) between model predictions and target values
function loss(x, y) #x is the input data (in this case values from dk), y is the target values
    predictions = model(x)[:NEE]  # Predict NEE from model. Model output is a named tuple, and the writing [:NEE] accesses specifically the NEE element of the tuple
    return sum((predictions .- y).^2) / length(y)  # Mean squared error
end

# Set up the optimizer
# the optimizer will adjust the model’s parameters (the ones detected by Functor) during training to minimize the loss.
opt = Flux.Adam(0.01)
# Adam ad gradient descent are two algorithms commonly used, and are just algorithms to find the gradient based on derivatives (while a Nelder-Mead,
# for example, is derivative-free and is therefore more suitable to less computationally intensive tasks but where the derivatives cannot be found).
# Other are available,for example:
# opt = Flux.RMSProp(learning_rate) #RMSProp (Root Mean Square Propagation)
# opt = Flux.AdaGrad(learning_rate) #AdaGrad (Adaptive Gradient)
# there also non-gradient based algorithms but neural networks are differentiable so no need for them, but in case of a complex
# hybrid model it might become needed. Genetic algorithms are also used for optimizing the NNet hyperparameters, but it costs computation.
# Simulated annealing could be an option for a complex parameter space of the function outside the NNet.

# Training loop (syntax is specifical to Flux.train!)
# The results (updated parameters) from each epoch are automatically carried over to the next because train! modifies directly
# the model's parameters. Flux.params(model) points to the model’s parameters, so any changes made by train! are applied to the model itself.
# The model retains its updated state after each training step, allowing subsequent epochs to build on the adjustments made in previous ones.
epochs = 100 # one epoch is a training iteration
for epoch in 1:epochs
    # general syntax for train!: Flux.train!(loss, params, data, opt), so loss function, parameters to train, data and optimizer
    Flux.train!(loss, Flux.params(model), data, opt) # Flux.params(model) returns a list of the model’s trainable parameters.
    println("Epoch $epoch complete. Loss: ", loss(parent(dk), target_data))
end

# Note: our Nnet is rather simple here, with very few parameters.
# We can do a grid search for optimizing that hyperparameter, or better try to learn how to use algorithms capable of optimizing
# hyperparameters as well


#This bloc checks if the trained model has achieved a loss below 0.1 on the dataset, set (arbitrarily) as criterion for successful training.
@testset "Model Training" begin
    @test loss(parent(dk), target_data) < 0.1  # Example test condition
end


println("Final Q10 values: ", model.Q10)



# Test what model_output produces for a given input
sample_input = Array(parent(dk[:, 1]))  # Convert to plain array

# Reshape sample_input to a 9x1 matrix
sample_input = reshape(sample_input, 9, 1)
println("Reshaped sample input size: ", size(sample_input))

output = model_output(sample_input)


# questios still open:
# - hyperparameter tuning (hopefully not needed because that's another level of complexity...)
# - How to query the Nnet for getting the importance of the features?




