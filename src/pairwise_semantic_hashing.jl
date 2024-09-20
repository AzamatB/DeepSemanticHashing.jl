using Distributions
using Lux
# using LuxCUDA
using Optimisers
using Random
using Zygote

# seeding
rng = Random.default_rng()
Random.seed!(rng, 0)

##############   Unsupervised Semantic Hashing with Pairwise Reconstruction   ##############
#
# See arxiv.org/abs/2007.00380 for details.

# model structure
struct PairRecSemanticHasher{D1,D2,DO,D3} <: LuxCore.AbstractLuxContainerLayer{(
    :dense_1, :dense_2, :dropout, :dense_3
)}
    dim_in::Int
    dim_encoding::Int

    dense_1::D1
    dense_2::D2
    dropout::DO
    dense_3::D3
end

function PairRecSemanticHasher(
    dim_in::Integer,
    dim_encoding::Integer,
    drop_prob::Real=0.2f0,
    dim_hidden_1::Integer=round(Int, range(start=dim_in, stop=dim_encoding, length=4)[2]),
    dim_hidden_2::Integer=round(Int, range(start=dim_in, stop=dim_encoding, length=4)[3])
)
    dense_1 = Dense(dim_in => dim_hidden_1, relu)
    dense_2 = Dense(dim_hidden_1 => dim_hidden_2, relu)
    dropout = Dropout(Float32(drop_prob))
    dense_3 = Dense(dim_hidden_2 => dim_encoding, σ)

    D1 = typeof(dense_1)
    D2 = typeof(dense_2)
    DO = typeof(dropout)
    D3 = typeof(dense_3)

    model = PairRecSemanticHasher{D1,D2,DO,D3}(
        dim_in, dim_encoding, dense_1, dense_2, dropout, dense_3
    )
    return model
end

function Lux.initialparameters(rng::AbstractRNG, model::PairRecSemanticHasher)
    importance_weights = rand(rng, Float32, model.dim_in)
    word_embedding = kaiming_uniform(rng, Float32, model.dim_in, model.dim_encoding)
    decoder_bias = Lux.init_linear_bias(rng, nothing, model.dim_encoding, model.dim_in)

    dense_1 = Lux.initialparameters(rng, model.dense_1)
    dense_2 = Lux.initialparameters(rng, model.dense_2)
    dropout = Lux.initialparameters(rng, model.dropout)
    dense_3 = Lux.initialparameters(rng, model.dense_3)

    params = (; importance_weights, dense_1, dense_2, dropout, dense_3, word_embedding, decoder_bias)
    return params
end

function Lux.initialstates(rng::AbstractRNG, model::PairRecSemanticHasher)
    dense_1 = Lux.initialstates(rng, model.dense_1)
    dense_2 = Lux.initialstates(rng, model.dense_2)
    dense_3 = Lux.initialstates(rng, model.dense_3)
    dropout = Lux.initialstates(rng, model.dropout)
    σ = 1.0f0

    state = (; dense_1, dense_2, dropout, dense_3, σ)
    return state
end

function Lux.parameterlength(model::PairRecSemanticHasher)
    # importance_weights: (dim_in × 1)
    # decoder_bias: (dim_in × 1)
    # word_embedding: (dim_in × dim_encoding)
    len = model.dim_in * (model.dim_encoding + 2)
    len += Lux.parameterlength(model.dense_1) # (dim_hidden_1 × dim_in) + (dim_hidden_1 × 1)
    len += Lux.parameterlength(model.dense_2) # (dim_hidden_2 × dim_hidden_1) + (dim_hidden_2 × 1)
    len += Lux.parameterlength(model.dropout) # 0
    len += Lux.parameterlength(model.dense_3) # (dim_encoding × dim_hidden_2) + (dim_encoding × 1)
    return len
end

function Lux.statelength(model::PairRecSemanticHasher)
    len = 1 # σ
    len += Lux.statelength(model.dense_1) # 0
    len += Lux.statelength(model.dense_2) # 0
    len += Lux.statelength(model.dropout) # 2
    len += Lux.statelength(model.dense_3) # 0
    return len
end

# TODO: move to utils.jl
function add_noise(x::AbstractArray, σ::Real, rng::AbstractRNG)
    𝓝 = Normal(0.0f0, Float32(σ))
    ε = rand(rng, 𝓝, size(x))
    return x + ε
end

# TODO: move to utils.jl
function sample_bernoulli_trials(probs::AbstractArray, rng::AbstractRNG)
    uniform_sample = rand(rng, Float32, size(probs))
    trials = (uniform_sample .< probs)
    return trials
end

# forward pass definition
function (model::PairRecSemanticHasher)(
    input::AbstractMatrix{<:Real}, params::NamedTuple, state::NamedTuple
)
    rng = state.dropout.rng
    importance_weights = params.importance_weights
    word_embedding = params.word_embedding
    decoder_bias = params.decoder_bias

    weighted_input = input .* importance_weights
    output_hidden_1, _ = model.dense_1(weighted_input, params.dense_1, state.dense_1)
    output_hidden_2, _ = model.dense_2(output_hidden_1, params.dense_2, state.dense_2)
    output_dropped, _ = model.dropout(output_hidden_2, params.dropout, state.dropout)
    encoding, _ = model.dense_3(output_dropped, params.dense_3, state.dense_3)

    hashcode = sample_bernoulli_trials(encoding, rng)
    noisy_hashcode = add_noise(hashcode, state.σ, rng)
    # (dim_in × dim_encoding) * (dim_encoding × batch_size) ≡ (dim_in × batch_size)
    projection = word_embedding * noisy_hashcode
    # (dim_in × batch_size) .* (dim_in × 1) .+ (dim_in × 1) ≡ (dim_in × batch_size)
    logits = @. projection * importance_weights + decoder_bias # (dim_in × batch_size)
    decoding = logsoftmax(logits; dims=1) # (dim_in × batch_size)

    output = (; encoding, decoding)
    return (output, state)
end


model = PairRecSemanticHasher(10, 2)
params, state = LuxCore.setup(rng, model)

# dummy input
input = rand(rng, Float32, 10, 7)

# run the model
output, state = Lux.apply(model, input, params, state)
