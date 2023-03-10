#include <math.h>

#include <vector>

#include "NeuMF.h"
#include "dataset.h"

int main() {
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    ProblemMode problem_mode = ProblemMode::REGRESSION;

    // data
    auto data = readAndSplitMovieLens20M("/home/helmut/Documents/Arbeit/Data/ml-20m/ratings.csv", 0.2, problem_mode,
                                         std::make_shared<torch::Device>(device));
    auto train_data = data.first.map(torch::data::transforms::Stack<>());
    auto test_data = data.second.map(torch::data::transforms::Stack<>());
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_data), 256);
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_data), 1);

    // hyper params
    const std::vector<int64_t> mlp_layers = {256, 128, 64, 32, 16, 8};
    const int64_t mf_dims = 30;
    const size_t num_epochs = 20;
    const double learning_rate = 0.01;

    // model
    NeuMF model(data.first.getNumOfUser() + 1, data.first.getNumOfItems() + 1, mlp_layers, problem_mode,
                std::make_shared<torch::Device>(device), mf_dims);
    model->to(device);

    // optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // training start time
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // training

    std::cout << "Started Training" << std::endl;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto& batch : *train_loader) {
            auto data = torch::transpose(batch.data, 0, 1);
            auto target = batch.target;

            auto output = model->forward(data[0], data[1]);

            torch::Tensor loss;

            if (problem_mode == ProblemMode::CLASSIFICATION)
                loss = torch::nn::functional::cross_entropy(output, target);
            else
                loss = torch::nn::functional::mse_loss(output, target);

            running_loss += loss.item<double>() * batch.data.size(0);

            if (problem_mode == ProblemMode::CLASSIFICATION) {
                auto prediction = output.argmax(1);
                num_correct += prediction.eq(target.argmax(1)).sum().item<int64_t>();
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

    std::cout << "Finished Training" << std::endl;

    // training end time
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Training took: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]"
              << std::endl;

    double rmse = 0.0;
    for (auto& batch : *test_loader) {
        auto data = torch::transpose(batch.data, 0, 1);
        auto target = batch.target;

        auto output = model->forward(data[0], data[1]);

        auto prediction = output.argmax();
        torch::Tensor loss;

        if (problem_mode == ProblemMode::CLASSIFICATION) {
            loss = torch::nn::functional::mse_loss(prediction.to(torch::kDouble), target.argmax().to(torch::kDouble));

        } else {
            loss = torch::nn::functional::mse_loss(output, target);
        }

        rmse += loss.item<double>();
    }
    std::cout << rmse << std::endl;
    // TODO: fix casting styles
    rmse /= double(data.second.size().value());
    rmse = sqrt(rmse);
    std::cout << "RMSE: " << rmse << std::endl;
    return 0;
}
