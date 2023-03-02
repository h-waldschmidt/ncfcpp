#include <c10/core/ScalarType.h>
#include <math.h>

#include <vector>

#include "NeuMF.h"
#include "dataset.h"

int main() {
    ProblemMode problem_mode = ProblemMode::REGRESSION;

    // data
    auto data = readAndSplitMovieLens1M("path", 0.2, problem_mode);
    auto train_data = data.first.map(torch::data::transforms::Stack<>());
    auto test_data = data.second.map(torch::data::transforms::Stack<>());
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_data), 128);
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_data), 1);

    // hyper params
    const std::vector<int64_t> mlp_layers = {256, 128, 64, 32, 16, 8};
    const int64_t mf_dims = 30;
    const size_t num_epochs = 20;
    const double learning_rate = 0.01;

    // model
    NeuMF model(data.first.getNumOfUser() + 1, data.first.getNumOfItems() + 1, mlp_layers, problem_mode, mf_dims);

    // optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // training
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto& batch : *train_loader) {
            auto data = torch::transpose(batch.data, 0, 1);
            auto output = model->forward(data[0], data[1]);

            torch::Tensor loss;

            if (problem_mode == ProblemMode::CLASSIFICATION)
                loss = torch::nn::functional::cross_entropy(output, batch.target);
            else
                loss = torch::nn::functional::mse_loss(output, batch.target);

            running_loss += loss.item<double>() * batch.data.size(0);

            if (problem_mode == ProblemMode::CLASSIFICATION) {
                auto prediction = output.argmax(1);
                num_correct += prediction.eq(batch.target.argmax(1)).sum().item<int64_t>();
            }

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
        auto prediction = output.argmax();
        torch::Tensor loss;

        if (problem_mode == ProblemMode::CLASSIFICATION) {
            loss = torch::nn::functional::mse_loss(prediction.to(torch::kDouble),
                                                   batch.target.argmax().to(torch::kDouble));

        } else {
            loss = torch::nn::functional::mse_loss(output, batch.target);
        }

        rmse += loss.item<double>();
    }
    std::cout << rmse << std::endl;
    rmse /= double(data.second.size().value());
    rmse = sqrt(rmse);
    std::cout << "RMSE: " << rmse << std::endl;
    return 0;
}
