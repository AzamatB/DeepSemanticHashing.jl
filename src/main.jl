using Base.Order: Forward
using ChainRules
using ChainRules: NoTangent
using DataStructures
using JLD2
using Lux
using LuxCUDA
using MLUtils
using Optimisers
using Printf
using Random
using Serialization
using StatsBase
using Zygote

CUDA.allowscalar(false)

const cpu = cpu_device()
const device = CUDA.functional() ? gpu_device() : cpu_device()

# seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

include("utilities.jl")

include("model.jl")

include("data_preparation.jl")


function train_model!(rng::AbstractRNG; num_epochs::Integer, learning_rate::Real = 0.001f0)
    # load datasets
    (dataset_train, dataset_val) = load_datasets(device)

    # construct the model
    dim_in = size(first(dataset_train), 1)
    model = PairRecSemanticHasher(dim_in)

    # move the model parameters & states to the GPU if possible
    params, states = LuxCore.setup(rng, model) |> device

    η = Float32(learning_rate)
    optimiser = Adam(η)
    ad_backend = AutoZygote()
    train_state = Training.TrainState(model, params, states, optimiser)
    data_train = DataLoader(dataset_train; batchsize=0, shuffle=true)

    display(model)
    states_val = Lux.testmode(train_state.states)
    loss_val_min = compute_dataset_loss(model, params, states_val, dataset_val)
    @printf "Validation loss before training:  %4.6f\n" loss_val_min
    @info "Training..."
    for epoch in 1:num_epochs
        # train the model
        loss_train = 0.0f0
        for inputs_batch in data_train
            (_, loss, _, train_state) = Training.single_train_step!(
                ad_backend, compute_loss, inputs_batch, train_state
            )
            batch_size = size(inputs_batch, 2)
            loss_train += loss / (batch_size^2)
            # @printf "Epoch [%3d]: Loss  %4.6f\n" epoch loss
        end
        loss_train /= length(dataset_train)
        @printf "Epoch [%3d]: Training Loss  %4.6f\n" epoch loss_train
        # validate the model
        states_val = Lux.testmode(train_state.states)
        loss_val = compute_dataset_loss(model, train_state.parameters, states_val, dataset_val)
        @printf "Epoch [%3d]: Validation loss  %4.6f\n" epoch loss_val
        if loss_val < loss_val_min
            loss_val_min = loss_val
            model_parameters = train_state.parameters |> cpu

            # delete previously saved model parameters
            rm("pretrained_weights", recursive=true)
            mkpath("pretrained_weights")

            # save the current model parameters
            jldsave(
                "pretrained_weights/model_weights_from_epoch_$(epoch)_$(model.dim_encoding)-bits.jld2";
                model_parameters
            )
        end
    end

    @info "Training completed."
    params_opt = train_state.parameters
    states_val = Lux.testmode(train_state.states)

    return model, params_opt, states_val
end


# set hyperparameters
η = 0.0004f0
num_epochs = 150

# train the model
@time (model, params, states) = train_model!(rng; num_epochs, learning_rate=η)
