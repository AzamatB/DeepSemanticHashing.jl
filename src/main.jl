# module DeepSemanticHashing

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

const device = CUDA.functional() ? gpu_device() : cpu_device()

# seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

include("utilities.jl")

include("model.jl")

include("data_preparation.jl")


function train_model!(
    model::PairRecSemanticHasher,
    params::NamedTuple,
    states::NamedTuple,
    dataset_train::AbstractVector{M},
    dataset_val::AbstractVector{M},
    dataset_test::AbstractVector{M};
    num_epochs::Integer,
    learning_rate::Real=0.001f0
) where {M<:DenseMatrix{Float32}}
    η = Float32(learning_rate)
    optimiser = Adam(η)
    ad_backend = AutoZygote()
    train_state = Training.TrainState(model, params, states, optimiser)
    data_train = DataLoader(dataset_train; batchsize=0, shuffle=true)

    display(model)
    states_val = Lux.testmode(train_state.states)
    loss_test = compute_dataset_loss(model, params, states_val, dataset_test)
    @printf "Test loss  %4.6f\n" loss_test

    loss_val_min = Inf32
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
            # save the model parameters
            model_parameters = train_state.parameters
            jldsave("pretrained_weights/model_weights_from_epoch_$(epoch).jld2"; model_parameters)
        end
    end
    @info "Training completed."

    params_opt = train_state.parameters
    loss_test = compute_dataset_loss(model, params_opt, states_val, dataset_test)
    @printf "Test loss  %4.6f\n" loss_test

    return model, params_opt, states_val
end


# load datasets
(dataset_train, dataset_val, dataset_test) = load_datasets(device)

# set hyperparameters
dim_in = size(first(dataset_train), 1)
dim_encoding = 512
β = 0.1f0
η = 0.0004f0
num_epochs = 100

# construct the model
model = PairRecSemanticHasher(dim_in, dim_encoding, β)
# move the model parameters & states to the GPU if possible
params, states = LuxCore.setup(rng, model) |> device

# train the model
@time (model, params, states) = train_model!(
    model,
    params,
    states,
    dataset_train,
    dataset_val,
    dataset_test;
    num_epochs,
    learning_rate=η
)

# # perform inference
# @info "Inference..."
# hashcode = encode(model, first(dataset_test), params)

# end # DeepSemanticHashing
