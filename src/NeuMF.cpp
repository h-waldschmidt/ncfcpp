#include "NeuMF.h"

#include <ATen/ops/where.h>

NeuMFImpl::NeuMFImpl(int64_t num_users, int64_t num_items, std::vector<int64_t> mlp_layers, ProblemMode problem_mode,
                     int64_t mf_dims)
    : m_mf_embedding_user(num_users, mf_dims),
      m_mf_embedding_item(num_items, mf_dims),
      m_mlp_embedding_user(num_users, mlp_layers[0] / 2),
      m_mlp_embedding_item(num_items, mlp_layers[0] / 2),
      m_prediction(nullptr),
      m_problem_mode(problem_mode) {
    register_module("mf_embedding_user", m_mf_embedding_user);
    register_module("mf_embedding_item", m_mf_embedding_item);
    register_module("mlp_embedding_user", m_mlp_embedding_user);
    register_module("mlp_embedding_item", m_mlp_embedding_item);

    register_module("mlp_layers", m_mlp_layers);
    for (int i = 1; i < mlp_layers.size(); i++) {
        auto cur_layer = torch::nn::Linear(mlp_layers[i - 1], mlp_layers[i]);
        auto activation_layer = torch::nn::ReLU();

        m_mlp_layers->push_back(cur_layer);
        m_mlp_layers->push_back(activation_layer);
    }

    if (problem_mode == ProblemMode::CLASSIFICATION) {
        torch::nn::Linear prediction(mf_dims + mlp_layers.back(), 5);
        m_prediction = prediction;
    } else {
        torch::nn::Linear prediction(mf_dims + mlp_layers.back(), 1);
        m_prediction = prediction;
    }

    register_module("prediction", m_prediction);
}

torch::Tensor NeuMFImpl::forward(torch::Tensor user_input, torch::Tensor item_input) {
    // Embedding layer
    torch::Tensor mf_embedding_user = m_mf_embedding_user->forward(user_input);
    torch::Tensor mf_embedding_item = m_mf_embedding_item->forward(item_input);
    torch::Tensor mlp_embedding_user = m_mlp_embedding_user->forward(user_input);
    torch::Tensor mlp_embedding_item = m_mlp_embedding_item->forward(item_input);
    mf_embedding_user.flatten();
    mf_embedding_item.flatten();
    mlp_embedding_user.flatten();
    mlp_embedding_item.flatten();

    // MF layer
    torch::Tensor mf_vector = torch::mul(mf_embedding_user, mf_embedding_item);

    // MLP layer
    torch::Tensor mlp_vector = torch::cat({mlp_embedding_user, mlp_embedding_item}, 1);
    mlp_vector = m_mlp_layers->forward(mlp_vector);

    // concatenate MF and MLP layer
    torch::Tensor output = torch::cat({mf_vector, mlp_vector}, 1);

    // final prediction
    output = m_prediction->forward(output);
    torch::Tensor problem_mode = torch::zeros(1, torch::kBool);
    problem_mode[0] = (m_problem_mode == ProblemMode::CLASSIFICATION);
    output = torch::where(problem_mode, torch::sigmoid(output), output);
    return output;
}
