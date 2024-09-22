# dummy input
input₁ = rand(rng, Float32, dim_in, batch_size)
input₂ = rand(rng, Float32, dim_in, batch_size)
input_pair = (input₁, input₂)

# run the model
(loss, states, _) = compute_loss(model, params, states, input_pair)

(loss, states, stats), back = Zygote.pullback(compute_loss, model, params, states, input_pair)
grads_t = back((one(loss), nothing, nothing))

Training.compute_gradients(ad_backend, compute_loss, input_pair, train_state)

############################################################################################

using PyCall

py"""
import pickle

def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""
load_pickle = py"load_pickle"

collection = load_pickle("/Users/aza/Projects/pairrec_data_code/data/reuters_collections")
(_, training, _, validation, testing, _, data_text_vect, labels, _, id2token) = collection
