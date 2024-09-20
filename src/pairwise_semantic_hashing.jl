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
    :dense₁, :dense₂, :dropout, :dense₃
)}
    dim_in::Int
    dim_encoding::Int

    dense₁::D1
    dense₂::D2
    dropout::DO
    dense₃::D3
end

function PairRecSemanticHasher(
    dim_in::Integer,
    dim_encoding::Integer,
    drop_prob::Real = 0.2f0,
    dim_hidden₁::Integer = round(Int, range(start=dim_in, stop=dim_encoding, length=4)[2]),
    dim_hidden₂::Integer = round(Int, range(start=dim_in, stop=dim_encoding, length=4)[3])
)
    dense₁ = Dense(dim_in => dim_hidden₁, relu)
    dense₂ = Dense(dim_hidden₁ => dim_hidden₂, relu)
    dropout = Dropout(Float32(drop_prob))
    dense₃ = Dense(dim_hidden₂ => dim_encoding, σ)

    D1 = typeof(dense₁)
    D2 = typeof(dense₂)
    DO = typeof(dropout)
    D3 = typeof(dense₃)

    model = PairRecSemanticHasher{D1,D2,DO,D3}(
        dim_in, dim_encoding, dense₁, dense₂, dropout, dense₃
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
    σ = 1.0f0

    state = (; dense₁, dense₂, dropout, dense₃, σ)
    return state
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
    len = 1 # σ
    len += Lux.statelength(model.dense₁) # 0
    len += Lux.statelength(model.dense₂) # 0
    len += Lux.statelength(model.dropout) # 2
    len += Lux.statelength(model.dense₃) # 0
    return len # 3
end

# TODO: move to utils.jl
function add_noise(x::AbstractVecOrMat{Bool}, σ::Float32, rng::AbstractRNG)
    𝓝 = Normal(0.0f0, σ)
    ε = rand(rng, 𝓝, size(x))
    return x + ε
end

# TODO: move to utils.jl
function sample_bernoulli_trials(probs::DenseVecOrMat{Float32}, rng::AbstractRNG)
    uniform_sample = rand(rng, Float32, size(probs))
    trials = (uniform_sample .< probs)
    return trials
end

# forward pass definition
function (model::PairRecSemanticHasher)(
    input::DenseVecOrMat{Float32}, params::NamedTuple, state::NamedTuple
)
    rng = state.dropout.rng
    importance_weights = params.importance_weights
    word_embedding = params.word_embedding
    decoder_bias = params.decoder_bias

    weighted_input = input .* importance_weights
    output_hidden₁, _ = model.dense₁(weighted_input, params.dense₁, state.dense₁)
    output_hidden₂, _ = model.dense₂(output_hidden₁, params.dense₂, state.dense₂)
    output_dropped, _ = model.dropout(output_hidden₂, params.dropout, state.dropout)
    encoding, _ = model.dense₃(output_dropped, params.dense₃, state.dense₃)

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

function loss(
    model::PairRecSemanticHasher,
    input_pair::NTuple{2,DenseVecOrMat{Float32}},
    params::NamedTuple,
    state::NamedTuple
)
    (input₁, input₂) = input_pair
    (encoding₁, decoding₁), _ = Lux.apply(model, input₁, params, state)
    (encoding₂, decoding₂), _ = Lux.apply(model, input₂, params, state)
    loss_kl₁ = kl_loss(encoding₁)
    loss_kl₂ = kl_loss(encoding₂)
    loss_recon₁ = reconstruction_loss(decoding₁, input₁)
    loss_recon₂ = reconstruction_loss(decoding₂, input₁)
    loss_total = loss_kl₁ + loss_kl₂ + loss_recon₁ + loss_recon₂
    return loss_total
end

# TODO: move to utils.jl
# Calculates Kullback-Leibler (KL) divergence between two multivariate Bernoulli
# distributions `probs` and q, where the distribution q is assumed to be such that
# qᵢ ∼ Bernoulli(0.5), ∀ i. This case has a closed form solution. For precise details, see:
# math.stackexchange.com/questions/2604566/kl-divergence-between-two-multivariate-bernoulli-distribution
function kl_loss(probs::DenseVecOrMat{Float32})
    ε = nextfloat(0.0f0)
    # add ε for numerical stability when calculating log()
    divergences = @. probs * log(2 * probs + ε) + (1 - probs) * log(2 * (1 - probs) + ε)
    loss_kl = sum(divergences)
    return loss_kl
end

# TODO: move to utils.jl
function reconstruction_loss(decoding::DenseVecOrMat{Float32}, target::DenseVecOrMat{Float32})
    # to maximize this, we will minimize it's negation
    masked_log_probs = @. decoding * (target > 0)
    loss_recon = -sum(masked_log_probs)
    return loss_recon
end


model = PairRecSemanticHasher(7, 3)
params, state = LuxCore.setup(rng, model)

# dummy input
input₁ = rand(rng, Float32, 7, 5)
input₂ = rand(rng, Float32, 7, 5)
input_pair = (input₁, input₂)

# run the model
l = loss(model, input_pair, params, state)
