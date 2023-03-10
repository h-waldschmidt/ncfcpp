#pragma once

#include <torch/torch.h>

#include <memory>
#include <vector>

#include "rating.h"

class MovieLens : public torch::data::datasets::Dataset<MovieLens> {
   public:
    // The mode in which the dataset is loaded
    enum Mode { TRAIN, TEST };

    explicit MovieLens(std::vector<MovieLensRating>& data, int64_t num_users, int64_t num_items,
                       ProblemMode problem_mode, std::shared_ptr<torch::Device> device, Mode mode = Mode::TRAIN);

    // Returns the `Example` at the given `index`.
    torch::data::Example<> get(size_t index) override;

    // Returns the size of the dataset.
    torch::optional<size_t> size() const override;

    // Returns true if this is the training subset of MovieLens.
    bool is_train() const noexcept;

    // Returns all user item pairs stacked into a single tensor.
    const torch::Tensor& getUserItemPairs() const;

    // Returns all ratings stacked into a single tensor.
    const torch::Tensor& getRatings() const;

    // Returns number of users
    const int64_t getNumOfUser() const;

    // Returns number of items
    const int64_t getNumOfItems() const;

    // Returns ProblemMode {Regression, Classification}
    const ProblemMode getProblemMode() const;

   private:
    std::shared_ptr<torch::Device> m_device;
    torch::Tensor m_user_item_pairs;
    torch::Tensor m_ratings;
    Mode m_mode;
    ProblemMode m_problem_mode;
    int64_t m_num_users;
    int64_t m_num_items;
};

/**
 * @brief reads, splits and creats two MovieLens Datasets for MovieLens 1M dataset
 *
 * @param data_path path to MovieLens file
 * @param test_size size of train/test split (e.g. 0.2 corresponds to 80% train
 * and 20% test data)
 * @return train and test data sets
 **/
std::pair<MovieLens, MovieLens> readAndSplitMovieLens1M(const std::string& data_path, double test_size,
                                                        ProblemMode problem_mode,
                                                        std::shared_ptr<torch::Device> device);

/**
 * @brief reads, splits and creats two MovieLens Datasets for MovieLens 20M dataset
 *
 * @param data_path path to MovieLens file
 * @param test_size size of train/test split (e.g. 0.2 corresponds to 80% train
 * and 20% test data)
 * @return train and test data sets
 **/
std::pair<MovieLens, MovieLens> readAndSplitMovieLens20M(const std::string& data_path, double test_size,
                                                         ProblemMode problem_mode,
                                                         std::shared_ptr<torch::Device> device);
