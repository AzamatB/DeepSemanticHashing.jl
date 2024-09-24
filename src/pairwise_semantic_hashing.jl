using ChainRules
using ChainRules: NoTangent
using Lux
using LuxCUDA
using Optimisers
using Printf
using Random
using Zygote

# seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

const device = CUDA.functional() ? gpu_device() : cpu_device()

##############   Unsupervised Semantic Hashing with Pairwise Reconstruction   ##############
#
# See arxiv.org/abs/2007.00380 for details.

# model structure
struct PairRecSemanticHasher{D₁,D₂,DO,D₃} <: LuxCore.AbstractLuxContainerLayer{(
    :dense₁, :dense₂, :dropout, :dense₃
)}
    dim_in::Int
    dim_encoding::Int
    dim_hidden₁::Int
    dim_hidden₂::Int

    dense₁::D₁
    dense₂::D₂
    dropout::DO
    dense₃::D₃
end

function PairRecSemanticHasher(
    dim_in::Integer,
    dim_encoding::Integer,
    drop_prob::Real = 0.2f0,
    dim_hidden₁::Integer = log_range(dim_in, dim_encoding, 4)[2],
    dim_hidden₂::Integer = log_range(dim_in, dim_encoding, 4)[3]
)
    dense₁ = Dense(dim_in => dim_hidden₁, relu)
    dense₂ = Dense(dim_hidden₁ => dim_hidden₂, relu)
    dropout = Dropout(Float32(drop_prob))
    dense₃ = Dense(dim_hidden₂ => dim_encoding, σ)

    D₁ = typeof(dense₁)
    D₂ = typeof(dense₂)
    DO = typeof(dropout)
    D₃ = typeof(dense₃)

    model = PairRecSemanticHasher{D₁,D₂,DO,D₃}(
        dim_in, dim_encoding, dim_hidden₁, dim_hidden₂, dense₁, dense₂, dropout, dense₃
    )
    return model
end

function Lux.initialparameters(rng::AbstractRNG, model::PairRecSemanticHasher)
    importance_weights = rand(rng, Float32, model.dim_in)
    word_embedding = kaiming_uniform(rng, Float32, model.dim_in, model.dim_encoding)
    decoder_bias = Lux.init_linear_bias(rng, nothing, model.dim_encoding, model.dim_in)

    dense₁ = Lux.initialparameters(rng, model.dense₁)
    dense₂ = Lux.initialparameters(rng, model.dense₂)
    dropout = Lux.initialparameters(rng, model.dropout)
    dense₃ = Lux.initialparameters(rng, model.dense₃)

    params = (; importance_weights, dense₁, dense₂, dropout, dense₃, word_embedding, decoder_bias)
    return params
end

function Lux.initialstates(rng::AbstractRNG, model::PairRecSemanticHasher)
    dense₁ = Lux.initialstates(rng, model.dense₁)
    dense₂ = Lux.initialstates(rng, model.dense₂)
    dense₃ = Lux.initialstates(rng, model.dense₃)
    dropout = Lux.initialstates(rng, model.dropout)
    λ = 1.0f0

    states = (; dense₁, dense₂, dropout, dense₃, λ)
    return states
end

function Lux.parameterlength(model::PairRecSemanticHasher)
    # importance_weights: (dim_in × 1)
    # decoder_bias: (dim_in × 1)
    # word_embedding: (dim_in × dim_encoding)
    len = model.dim_in * (model.dim_encoding + 2)
    len += Lux.parameterlength(model.dense₁) # (dim_hidden₁ × dim_in) + (dim_hidden₁ × 1)
    len += Lux.parameterlength(model.dense₂) # (dim_hidden₂ × dim_hidden₁) + (dim_hidden₂ × 1)
    len += Lux.parameterlength(model.dropout) # 0
    len += Lux.parameterlength(model.dense₃) # (dim_encoding × dim_hidden₂) + (dim_encoding × 1)
    return len
end

function Lux.statelength(model::PairRecSemanticHasher)
    len = 1 # λ
    len += Lux.statelength(model.dense₁) # 0
    len += Lux.statelength(model.dense₂) # 0
    len += Lux.statelength(model.dropout) # 2
    len += Lux.statelength(model.dense₃) # 0
    return len # 3
end

# TODO: move to utils.jl
function log_range(start::Real, stop::Real, len::Integer)
    exp_start = log(start)
    exp_stop = log(stop)
    exp_rng = range(exp_start, exp_stop; length=len)
    rng = @. round(Int, exp(exp_rng))
    return rng
end

# TODO: move to utils.jl
function add_noise(x::AbstractMatrix{Bool}, λ::Float32, rng::AbstractRNG)
    # add small white noise to the encoding
    ε = λ * randn(rng, Float32, size(x))
    return x + ε
end

# TODO: move to utils.jl
function sample_bernoulli(probs::DenseMatrix{Float32}, rng::AbstractRNG)
    # sample (multivariate) Bernoulli distribution specified by success probabilities
    # `probs`
    uniform_sample = rand(rng, Float32, size(probs))
    trials = (uniform_sample .< probs)
    return trials
end

# TODO: move to utils.jl
# straight-through estimator for the gradient of `add_noise` function, i.e. we are assuming
# that it behaves as an identity function (λ = 0) for the purpose of gradient computation.
# function ChainRules.rrule(
#     ::typeof(add_noise), x::AbstractMatrix{Bool}, λ::Float32, rng::AbstractRNG
# )
#     function identity_pullback(ȳ)
#         return (NoTangent(), ȳ, NoTangent(), NoTangent())
#     end
#     return (add_noise(x, λ, rng), identity_pullback)
# end

# TODO: move to utils.jl
# `sample_bernoulli(..)` is not differentiable in a strict sense, so to work around this we
# define a straight-through estimator for its gradient, i.e. we are assuming that it behaves
# as an identity function for the purpose of gradient computation.
# See arxiv.org/abs/1308.3432 for some theoretical and empirical justifications behind this.
function ChainRules.rrule(
    ::typeof(sample_bernoulli), probs::DenseMatrix{Float32}, rng::AbstractRNG
)
    function identity_pullback(ȳ)
        return (NoTangent(), ȳ, NoTangent())
    end
    return (sample_bernoulli(probs, rng), identity_pullback)
end

# TODO: move to utils.jl
function decay_noise(states::NamedTuple)
    λ = max(states.λ - 1.0f-6, 0.0f0)
    states = (; states.dense₁, states.dense₂, states.dropout, states.dense₃, λ)
    return states
end


# forward pass definition
function (model::PairRecSemanticHasher)(
    input::DenseMatrix{Float32}, params::NamedTuple, states::NamedTuple
)
    rng = states.dropout.rng
    importance_weights = params.importance_weights
    word_embedding = params.word_embedding
    decoder_bias = params.decoder_bias

    weighted_input = input .* importance_weights
    output_hidden₁, _ = model.dense₁(weighted_input, params.dense₁, states.dense₁)
    output_hidden₂, _ = model.dense₂(output_hidden₁, params.dense₂, states.dense₂)
    output_dropped, _ = model.dropout(output_hidden₂, params.dropout, states.dropout)
    encoding, _ = model.dense₃(output_dropped, params.dense₃, states.dense₃)

    hashcode = sample_bernoulli(encoding, rng)
    noisy_hashcode = add_noise(hashcode, states.λ, rng)
    # (dim_in × dim_encoding) * (dim_encoding × batch_size) ≡ (dim_in × batch_size)
    projection = word_embedding * noisy_hashcode
    # (dim_in × batch_size) .* (dim_in × 1) .+ (dim_in × 1) ≡ (dim_in × batch_size)
    logits = @. projection * importance_weights + decoder_bias # (dim_in × batch_size)
    decoding = logsoftmax(logits; dims=1) # (dim_in × batch_size)

    output = (; encoding, decoding)
    states = decay_noise(states)
    return (output, states)
end

function encode(
    model::PairRecSemanticHasher,
    input::DenseVecOrMat{<:Real},
    params::NamedTuple
)
    empty_state = (;)
    dropout_state = (; rng=Random.default_rng(), training=Val(false))
    importance_weights = params.importance_weights

    weighted_input = input .* importance_weights
    output_hidden₁, _ = model.dense₁(weighted_input, params.dense₁, empty_state)
    output_hidden₂, _ = model.dense₂(output_hidden₁, params.dense₂, empty_state)
    output_dropped, _ = model.dropout(output_hidden₂, params.dropout, dropout_state)
    probs, _ = model.dense₃(output_dropped, params.dense₃, empty_state)
    # greedily choose most probable bits according to the (multivariate) Bernoulli
    # distribution specified by success probabilities `probs`
    hashcode = round.(Bool, probs)
    return hashcode
end

function compute_loss(
    model::PairRecSemanticHasher,
    params::NamedTuple,
    states::NamedTuple,
    inputs::DenseMatrix{Float32}
)
    (encodings, decodings), states = Lux.apply(model, inputs, params, states)
    loss_kl = compute_kl_loss(encodings)
    loss_recon = -sum(inputs'decodings)
    loss_total = loss_kl + loss_recon
    return (loss_total, states, (;))
end

# TODO: move to utils.jl
# Calculates Kullback-Leibler (KL) divergence between two multivariate Bernoulli
# distributions `probs` and q, where the distribution q is assumed to be such that
# qᵢ ∼ Bernoulli(0.5), ∀ i. This case has a closed form solution. For precise details, see:
# math.stackexchange.com/questions/2604566/kl-divergence-between-two-multivariate-bernoulli-distribution
function compute_kl_loss(probs::DenseMatrix{Float32})
    ε = 1.0f-45 # the output of nextfloat(zero(Float32))
    # add small ε to avoid passing 0 to log(..)
    divergences = @. probs * log(2 * probs + ε) + (1 - probs) * log(2 * (1 - probs) + ε)
    loss_kl = sum(divergences)
    return loss_kl
end


function train_model!(
    model::PairRecSemanticHasher,
    params::NamedTuple,
    states::NamedTuple,
    data_train,
    data_val;
    num_epochs::Integer,
    learning_rate::Real = 0.001f0
)
    η = Float32(learning_rate)
    optimiser = Adam(η)
    ad_backend = AutoZygote()
    train_state = Training.TrainState(model, params, states, optimiser)

    local states_val
    for epoch in 1:num_epochs
        # train the model
        for input_batch in data_train
            (_, loss, _, train_state) = Training.single_train_step!(
                ad_backend, compute_loss, input_batch, train_state
            )
            @printf "Epoch [%3d]: Loss   %4.5f\n" epoch loss
        end
        # validate the model
        states_val = Lux.testmode(train_state.states)
        (loss, _, _) = compute_loss(model, train_state.parameters, states_val, data_val)
        @printf "\n\nValidation: Loss   %4.5f\n\n\n" loss
    end
    return model, train_state.parameters, states_val
end

dim_in = 23834
dim_encoding = 512
batch_size = 32
num_epochs = 3
η = 0.001f0

model = PairRecSemanticHasher(dim_in, dim_encoding)
params, states = LuxCore.setup(rng, model) |> device
rng = states.dropout.rng

############################################################################################
include("data_preparation.jl")
data_train = device(load_dataset(rng))
data_val = first(data_train)
############################################################################################
@info "Training..."
@time (model, params, states) = train_model!(
    model, params, states, data_train, data_val; num_epochs, learning_rate = η
)

@info "Inference..."
@show encode(model, data_val, params)
