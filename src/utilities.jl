# Utility functions

function log_range(start::Real, stop::Real, len::Integer)
    exp_start = log(start)
    exp_stop = log(stop)
    exp_rng = range(exp_start, exp_stop; length=len)
    rng = @. round(Int, exp(exp_rng))
    return rng
end

function add_noise(x::AbstractMatrix{Bool}, λ::Float32, rng::AbstractRNG)
    # add small white noise to the encoding
    ε = λ * randn(rng, Float32, size(x))
    return x + ε
end

function sample_bernoulli(probs::DenseMatrix{Float32}, rng::AbstractRNG)
    # sample (multivariate) Bernoulli distribution specified by success probabilities
    # `probs`
    uniform_sample = rand(rng, Float32, size(probs))
    trials = (uniform_sample .< probs)
    return trials
end

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

function decay_noise(states::NamedTuple)
    λ = max(states.λ - 1.0f-6, 0.0f0)
    states = (; states.dense₁, states.dense₂, states.dropout, states.dense₃, λ)
    return states
end

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
