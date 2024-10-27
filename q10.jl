
using Flux #loading the module for NNets training and definition


# Define helper function to create RUE and Rb models with Nnets
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
    # - `affine`: Determines if the BatchNorm layer includes learnable "affine" (whatever that means) parameters (defaults to true).

    return Flux.Chain(
        
        # Batch Normalization layer on input features
        # Normalizes the input data for faster convergence, with an option for learnable parameters.
        # affine comes from the main function container, default is true. If true, normalization includes "learnable affine transformation parameters", which applies scaling and shifting. 
        # this causes the main function to have two more hyperparameters, scaling and bias, for the normalization
        BatchNorm(in_dim, affine=affine),
        
        # what is a dense layer:  a dense layer is a layer in a neural network where every neuron is connected to every neuron in the preceding layer. 
        # For some reasons Reichstein decided for dense layers.

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


# Define model structure
struct FluxPartModel_Q10
    RUE_chain::Flux.Chain #type `Flux.Chain`, a sequence of layers. This contains the actual Nnet and it is an object specific to Fluz. Layers are not the same structure.
                          # for example we could have a (dense) layer with 5 inputs, 6 outputs, and a certain activator, and another with 2 inputs, 8 outputs and another activator
    Rb_chain::Flux.Chain

    Q10::Vector{Float32} #vector of 32-bit floating-point numbers (Float32) (in this case representing Q10 values)
    # ┌────────────────────────────────────────────────────────────────┐
    # │ !!! NOTE: Modified from original code, where Q10 was undefined.│
    # │ Julia is a dynamically typed language.                         │
    # │ It accepts type-flexible fields, but without specifying        │
    # │ `Q10::Vector{Float32}`, Flux’s `@functor` would not detect     │
    # │ `Q10` as a trainable parameter for `train!`.                   │
    # └────────────────────────────────────────────────────────────────┘
end

# This function initializes a new FluxPartModel_Q10 object with specific values for Q10 and neurons.
function FluxPartModel_Q10(; Q10=[1.5f0], neurons=15)
    RUE_chain = Dense_RUE_Rb(2; neurons)    # Create the RUE_chain neural network model. Assume 2 predictors for simplicity.
    Rb_chain = Dense_RUE_Rb(2; neurons)    #  Create the Rb_chain neural network model. Assume 2 predictors for simplicity.

    # Returns a new FluxPartModel_Q10 object initialized with the RUE_chain, Rb_chain, and Q10 vector.
    # The Q10 parameter here has a default value of [1.5f0], which is a 32-bit floating-point number (Float32).
    return FluxPartModel_Q10(RUE_chain, Rb_chain, Q10) #Q10 at this stage is just the initial value of the optimization
end


# Model inference method, here we are adding the process-based function part. 
# this part enables the FluxPartModel_Q10 to be used as a function
function (m::FluxPartModel_Q10)(dk) # this syntax overloads this part onto the previously defined FluxPartModel_Q10 structure, defining it as m 

    # The function relies on dk as input, which is a 9-row by 1,000-column matrix where:
    # Each row is labeled according to row_keys and represents a specific predictor or derived variable.
    # Each column represents an individual observation, containing values for each predictor and derived variable in that row.

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


    # Extract input data (from dk) for RUE_chain and Rb_chain by fixed index positions. These are the predictors for the Nnets.
    # RUE_input and Rb_input are assumed to use specific columns (or features) of `dk`
    RUE_input = dk[[1, 2], :]   # whatever these predictors are, we assume RUE is influenced by the ones in position 1 and 2
    Rb_input = dk[[3, 4], :]    # whatever these predictors are, we assume RUE is influenced by the ones in position 1 and 2

    # Print shapes before transposing
    #println("RUE_input shape before transpose: ", size(RUE_input))
    #println("Rb_input shape before transpose: ", size(Rb_input))

    # Pass the inputs through RUE_chain and Rb_chain NNet models
        # `m.RUE_chain(RUE_input)` produces RUE predictions, taking the first output after transposing
        RUE = m.RUE_chain(RUE_input)[:, 1]  #RUE (Radiative Use Efficiency)
        println("RUE output: ", RUE) #debug line to print the output
        # `m.Rb_chain(Rb_input)` produces Rb predictions, scaled by 100.0f0 to represent baseline respiration
        Rb = 100.0f0 * m.Rb_chain(Rb_input)[:, 1]  #Rb (Respiration or Basal Respiration)
        println("Rb output: ", Rb) #debug line to print the output

    # Access additional environmental variables by their fixed indices in `dk`
    sw_in = dk[5, :] 
    ta = dk[1, :] # this one is used both in the Nnet and in the process-based part

    # Calculate GPP and Reco
    GPP = sw_in .* RUE ./ 12.011f0
    Reco = Rb .* m.Q10[1] .^ (0.1f0 * (ta .- 15.0f0))
    #println("GPP: ", GPP) #debug line to print the output
    #println("Reco: ", Reco) #debug line to print the output
    #println("sw_in: ", sw_in)  # debug line to print the output
    #println("ta: ", ta)        # debug line to print the output
    #println("Q10: ", m.Q10)    # debug line to print the output
    


    # Return the result as a named tuple, where NEE (Net Ecosystem Exchange) is the difference between Reco and GPP
    # A named tuple is  immutable collection with labeled elements, accessible by their names.
    return (; NEE = Reco - GPP)
end

# Allows the model to work with Flux's `train!` function
# The @functor macro tells Flux how to access and update the fields in FluxPartModel_Q10 (e.g., RUE_chain, Rb_chain, and Q10) during training. 
# This is important because train! relies on accessing the model's parameters to compute gradients and perform updates.
# Specifically, @functor marks which fields in the struct should be treated as model parameters, allowing them to be optimized by Flux’s training routines.
Flux.@functor FluxPartModel_Q10
