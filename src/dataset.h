#pragma once

#include <torch/torch.h>

#include <vector>

// defines a rating given by the MovieLens dataset
struct MovieLensRating {
    int itemID;
    int userID;
    double rating;
};

struct MovieLens : torch::data::datasets::Dataset<MovieLens> {
   public:
    // The mode in which the dataset is loaded
    enum Mode { TRAIN, TEST };

    explicit MovieLens(std::vector<MovieLensRating>& data, Mode mode = Mode::TRAIN);

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

   private:
    torch::Tensor m_user_item_pairs;
    torch::Tensor m_ratings;
    Mode m_mode;
};

/**
 * @brief
 *
 * @param data_path path to MovieLens file
 * @param test_size size of train/test split (e.g. 0.2 corresponds to 80% train
 * and 20% test data)
 * @return train and test data sets
 **/
std::pair<MovieLens, MovieLens> readAndSplitMovieLens(const std::string& data_path, double test_size);