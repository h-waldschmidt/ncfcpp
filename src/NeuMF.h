#pragma once

#include <torch/torch.h>

#include <vector>

class NeuMFImpl : public torch::nn::Module {
   public:
    NeuMFImpl(int64_t num_users, int64_t num_items, std::vector<int64_t> mlp_layers, int64_t mf_dims = 10,
              int64_t output_dims = 5);
    torch::Tensor forward(torch::Tensor user_input, torch::Tensor item_input);

   private:
    torch::nn::Embedding m_mf_embedding_user;
    torch::nn::Embedding m_mf_embedding_item;
    torch::nn::Embedding m_mlp_embedding_user;
    torch::nn::Embedding m_mlp_embedding_item;

    torch::nn::Sequential m_mlp_layers;

    torch::nn::Linear m_prediction;
};

TORCH_MODULE(NeuMF);