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
    dim_encoding::Integer = 16,
    dim_hidden₁::Integer = ceil(Int, range(dim_in, dim_encoding; length = 4)[2]),
    dim_hidden₂::Integer = ceil(Int, range(dim_in, dim_encoding; length = 4)[3]),
    drop_prob::Real = 0.1f0
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
    dense₃ = Lux.initialparameters(rng, model.dense₃)

    params = (; importance_weights, dense₁, dense₂, dense₃, word_embedding, decoder_bias)
    return params
end

function Lux.initialstates(rng::AbstractRNG, model::PairRecSemanticHasher)
    dropout = Lux.initialstates(rng, model.dropout)
    λ = 1.0f0
    states = (; dropout, λ)
    return states
end

function Lux.parameterlength(model::PairRecSemanticHasher)
    # importance_weights: (dim_in × 1)
    # decoder_bias: (dim_in × 1)
    # word_embedding: (dim_in × dim_encoding)
    len = model.dim_in * (model.dim_encoding + 2)
    len += Lux.parameterlength(model.dense₁) # (dim_hidden₁ × dim_in) + (dim_hidden₁ × 1)
    len += Lux.parameterlength(model.dense₂) # (dim_hidden₂ × dim_hidden₁) + (dim_hidden₂ × 1)
    len += Lux.parameterlength(model.dense₃) # (dim_encoding × dim_hidden₂) + (dim_encoding × 1)
    return len
end

function Lux.statelength(model::PairRecSemanticHasher)
    len = 1 # λ
    len += Lux.statelength(model.dropout) # 2
    return len # 3
end

# forward pass definition
function (model::PairRecSemanticHasher)(
    input::DenseMatrix{Float32}, params::NamedTuple, states::NamedTuple
)
    rng = states.dropout.rng
    importance_weights = params.importance_weights
    word_embedding = params.word_embedding
    decoder_bias = params.decoder_bias
    # encoding stage
    weighted_input = input .* importance_weights
    output_hidden₁, _ = model.dense₁(weighted_input, params.dense₁, (;))
    output_hidden₂, _ = model.dense₂(output_hidden₁, params.dense₂, (;))
    output_dropped, _ = model.dropout(output_hidden₂, (;), states.dropout)
    encoding, _ = model.dense₃(output_dropped, params.dense₃, (;))
    hashcode = sample_bernoulli(encoding, rng)

    # decoding stage
    noisy_hashcode = add_noise(hashcode, states.λ, rng)
    # (dim_in × dim_encoding) * (dim_encoding × batch_size) ≡ (dim_in × batch_size)
    projection = word_embedding * noisy_hashcode
    # (dim_in × batch_size) .* (dim_in × 1) .+ (dim_in × 1) ≡ (dim_in × batch_size)
    logits = @. projection * importance_weights + decoder_bias # (dim_in × batch_size)
    decoding = logsoftmax(logits; dims=1) # (dim_in × batch_size)

    # decay noise
    λ = max(states.λ - 1.0f-6, 0.0f0)
    states = (; states.dropout, λ)

    return (decoding, states)
end

function encode(
    model::PairRecSemanticHasher,
    input::DenseVecOrMat{<:Real},
    params::NamedTuple
)
    weighted_input = input .* params.importance_weights
    output_hidden₁, _ = model.dense₁(weighted_input, params.dense₁, (;))
    output_hidden₂, _ = model.dense₂(output_hidden₁, params.dense₂, (;))
    probs, _ = model.dense₃(output_hidden₂, params.dense₃, (;))
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
    decodings, states = Lux.apply(model, inputs, params, states)
    loss = -sum(inputs'decodings)
    return (loss, states, (;))
end

function compute_dataset_loss(
    model::PairRecSemanticHasher,
    params::NamedTuple,
    states::NamedTuple,
    dataset::AbstractVector{<:DenseMatrix{Float32}}
)
    loss_total = 0.0f0
    for data_batch in dataset
        (loss, _, _) = compute_loss(model, params, states, data_batch)
        batch_size = size(data_batch, 2)
        loss_total += loss / (batch_size^2)
    end
    # a mean of means
    loss_total /= length(dataset)
    return loss_total
end
