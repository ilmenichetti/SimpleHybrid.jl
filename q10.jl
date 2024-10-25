using Flux

# Define helper function to create RUE and Rb models
function Dense_RUE_Rb(in_dim; neurons=15, out_dim=1, affine=true)
    # This function initializes a neural network with the following architecture:
    # - A Batch Normalization layer on the input features
    # - A Dense (fully connected) hidden layer with ReLU activation
    # - A Dense output layer with a sigmoid activation function (σ)
    #
    # Parameters:
    # - `in_dim`: (integer) refers to the dimensions of the covariates as input for the Nnet, could be for example 1 for a single variable
    # - `neurons`: Number of neurons in the hidden layer (defaults to 15).
    # - `out_dim`: Number of output neurons (defaults to 1).
    # - `affine`: Determines if the BatchNorm layer includes learnable affine parameters (defaults to true).

    return Flux.Chain(
        # Batch Normalization layer on input features
        # Normalizes the input data for faster convergence, with an option for learnable parameters.
        # affine comes from the main function container, default is true. If true, normalization includes "learnable affine transformation parameters", which applies scaling and shifting. 
        # this causes the main function to have two more hyperparameters, scaling and bias, for the normalization
        BatchNorm(in_dim, affine=affine),
        
        # dense layers:  a dense layer is a layer in a neural network where every neuron is connected to every neuron in the preceding layer. For some reasons Reichstein decided for dense layers.

        # Hidden Dense layer:
        # This layer connects `in_dim` inputs to `neurons` outputs, applying the ReLU activation function.
        # ReLU (Rectified Linear Unit) is often used for hidden layers due to its simplicity and effectiveness.
        # in_dim => neurons in Julia creates a pairing. It expresses the transformations from inputs to neurons, which can anyway be independent integers (declared in the input of the main function)
        # this pairing apply weights and biases as neeeded to obtain a Nnet with the defined charachteristics in terms of pairing
        # ReLU function is ReLU(x)=max(0,x), so it means basically "only positive values otherwise zero", and it is a basic activation function of the Nnet
        # an activation function is a component of each fundamental node and determines if a neuron is activated ("fired") or not. Without activation functions, 
        # a neural network would behave just like a very complicated linear transformation.
        Dense(in_dim => neurons, relu),
        
        # Output Dense layer:
        # This layer connects `neurons` inputs from the hidden layer to `out_dim` outputs, applying the sigmoid (σ) activation.
        # The sigmoid activation function is typically used for outputs where a probability or normalized output is desired.
        Dense(neurons => out_dim, σ)
    )
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
