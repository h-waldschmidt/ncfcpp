#pragma once

#include <torch/torch.h>

#include <vector>

// defines a rating given by the MovieLens dataset
struct MovieLensRating {
    int itemID;
    int userID;
    double rating;
};

class MovieLens : torch::data::datasets::Dataset<MovieLens> {
   public:
    // The mode in which the dataset is loaded
    enum Mode { TRAIN, TEST };

    explicit MovieLens(std::vector<MovieLensRating>& data, int64_t num_users, int64_t num_items, int64_t ouput_dims = 5,
                       Mode mode = Mode::TRAIN);

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
    int64_t m_num_users;
    int64_t m_num_items;
};

/**
 * @brief
 *
 * @param data_path path to MovieLens file
 * @param test_size size of train/test split (e.g. 0.2 corresponds to 80% train
 * and 20% test data)
 * @return train and test data sets
 **/
std::pair<MovieLens, MovieLens> readAndSplitMovieLens(const std::string& data_path, double test_size,
                                                      int64_t output_dims = 5);