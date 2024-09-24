# module DeepSemanticHashing

using Base.Order: Forward
using ChainRules
using ChainRules: NoTangent
using DataStructures
using Lux
using LuxCUDA
using MLUtils
using Optimisers
using Printf
using Random
using Serialization
using StatsBase
using Zygote

const device = CUDA.functional() ? gpu_device() : cpu_device()

# seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

include("utilities.jl")

include("model.jl")

include("data_preparation.jl")


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

# perform inference
@info "Inference..."
hashcode = encode(model, first(dataset_test), params)

# end # DeepSemanticHashing
