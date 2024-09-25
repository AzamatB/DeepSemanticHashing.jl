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
    β::Float32 # weight of the KL regularizer in the loss function

    dense₁::D₁
    dense₂::D₂
    dropout::DO
    dense₃::D₃
end

function PairRecSemanticHasher(
    dim_in::Integer,
    dim_encoding::Integer,
    β::Real,
    drop_prob::Real = 0.1f0,
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
        dim_in, dim_encoding, dim_hidden₁, dim_hidden₂, Float32(β), dense₁, dense₂, dropout, dense₃
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
    output_hidden₁, _ = model.dense₁(weighted_input, params.dense₁, states.dense₁)
    output_hidden₂, _ = model.dense₂(output_hidden₁, params.dense₂, states.dense₂)
    output_dropped, _ = model.dropout(output_hidden₂, params.dropout, states.dropout)
    encoding, _ = model.dense₃(output_dropped, params.dense₃, states.dense₃)
    hashcode = sample_bernoulli(encoding, rng)

    # decoding stage
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
    loss_total = loss_recon + model.β * loss_kl
    return (loss_total, states, (;))
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
    return loss_total
end

function train_model!(
    model::PairRecSemanticHasher,
    params::NamedTuple,
    states::NamedTuple,
    dataset_train::AbstractVector{M},
    dataset_val::AbstractVector{M},
    dataset_test::AbstractVector{M};
    num_epochs::Integer,
    learning_rate::Real = 0.001f0
) where {M<:DenseMatrix{Float32}}
    η = Float32(learning_rate)
    optimiser = Adam(η)
    ad_backend = AutoZygote()
    train_state = Training.TrainState(model, params, states, optimiser)
    data_train = DataLoader(dataset_train; batchsize=0, shuffle=true)

    states_val = Lux.testmode(train_state.states)
    loss_test = compute_dataset_loss(model, params, states_val, dataset_test)
    @printf "Test loss  %4.6f\n" loss_test

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
        @printf "Epoch [%3d]: Training Loss  %4.6f\n" epoch loss_train
        # validate the model
        states_val = Lux.testmode(train_state.states)
        loss_val = compute_dataset_loss(model, train_state.parameters, states_val, dataset_val)
        @printf "Epoch [%3d]: Validation loss  %4.6f\n" epoch loss_val
    end
    @info "Training completed."

    params_opt = train_state.parameters
    loss_test = compute_dataset_loss(model, params_opt, states_val, dataset_test)
    @printf "Test loss  %4.6f\n" loss_test

    return model, params_opt, states_val
end
