#include <math.h>

#include <vector>

#include "NeuMF.h"
#include "dataset.h"

int main() {
    // data
    auto data = readAndSplitMovieLens("../data/ml-1m/ratings.dat", 0.2);
    auto train_data = data.first.map(torch::data::transforms::Stack<>());
    auto test_data = data.second.map(torch::data::transforms::Stack<>());
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_data), 256);
    auto test_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_data), 256);

    // hyper params
    const std::vector<int64_t> mlp_layers = {64, 32, 16, 8};
    const int64_t mf_dims = 10;
    const int64_t output_dims = 1;
    const size_t num_epochs = 20;
    const double learning_rate = 0.02;

    // model
    NeuMF model(data.first.getNumOfUser() + 1, data.first.getNumOfItems() + 1, mlp_layers, mf_dims, output_dims);

    // optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // training
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto& batch : *train_loader) {
            auto data = torch::transpose(batch.data, 0, 1);
            auto output = model->forward(data[0], data[1]);
            auto loss = torch::nn::functional::mse_loss(output, batch.target);

            running_loss += loss.item<double>() * batch.data.size(0);

            auto prediction = output.argmax() + 1;
            num_correct += prediction.eq(batch.target).sum().item<int64_t>();

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / double(data.first.size().value());
        auto accuracy = static_cast<double>(num_correct) / double(data.first.size().value());

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: " << sample_mean_loss
                  << ", Accuracy: " << accuracy << '\n';
    }

    double rmse = 0.0;
    for (auto& batch : *test_loader) {
        auto data = torch::transpose(batch.data, 0, 1);
        auto output = model->forward(data[0], data[1]);
        auto loss = torch::nn::functional::mse_loss(output, batch.target);
        rmse += loss.item<double>() * batch.data.size(0);
    }
    std::cout << rmse << std::endl;
    rmse /= double(data.second.size().value());
    rmse = sqrt(rmse);
    std::cout << "RMSE: " << rmse << std::endl;
    return 0;
}
