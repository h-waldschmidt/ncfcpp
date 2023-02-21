#include "NeuMF.h"

NeuMFImpl::NeuMFImpl(int64_t num_users, int64_t num_items, std::vector<int64_t> mlp_layers, int64_t mf_dims = 10,
                     int64_t output_dims = 5)
    : m_mf_embedding_user(num_users, mf_dims),
      m_mf_embedding_item(num_items, mf_dims),
      m_mlp_embedding_user(num_users, m_mlp_layers[0]),
      m_mlp_embedding_item(num_items, mlp_layers[0]),
      m_prediction(mf_dims + mlp_layers.back(), output_dims) {
    register_module("mf_embedding_user", m_mf_embedding_user);
    register_module("mf_embedding_item", m_mf_embedding_item);
    register_module("mlp_embedding_user", m_mlp_embedding_user);
    register_module("mlp_embedding_item", m_mlp_embedding_item);

    register_module("mlp_laysers", m_mlp_layers);
    for (int i = 1; i < mlp_layers.size(); i++) {
        auto cur_layer = torch::nn::Sequential(mlp_layers[i - 1], mlp_layers[i]);
        auto activation_layer = torch::nn::Sigmoid(mlp_layers[i]);

        m_mlp_layers->push_back(cur_layer);
        m_mlp_layers->push_back(activation_layer);
    }

    register_module("prediction", m_prediction);
}

torch::Tensor NeuMFImpl::forward(torch::Tensor input) {
    // Embedding layer
    torch::Tensor mf_embedding_user = m_mf_embedding_user->forward(input[1]);
    torch::Tensor mf_embedding_item = m_mf_embedding_item->forward(input[0]);
    torch::Tensor mlp_embedding_user = m_mlp_embedding_user->forward(input[1]);
    torch::Tensor mlp_embedding_item = m_mlp_embedding_item->forward(input[0]);
    mf_embedding_user.flatten();
    mf_embedding_item.flatten();
    mlp_embedding_user.flatten();
    mlp_embedding_item.flatten();

    // MF layer
    torch::Tensor mf_vector = torch::mul(mf_embedding_user, mf_embedding_item);

    // MLP layer
    torch::Tensor mlp_vector = torch::cat({mlp_embedding_user, mlp_embedding_item});
    mlp_vector = m_mlp_layers->forward(mlp_vector);

    // concatenate MF and MLP layer
    torch::Tensor output = torch::cat({mf_vector, mlp_vector});

    // final prediction
    output = m_prediction->forward(output);
    return output;
}