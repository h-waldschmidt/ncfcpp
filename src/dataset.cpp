#include "dataset.h"

/**
 * @brief
 *
 * @param data_path path to MovieLens file
 * @param test_size size of train/test split (e.g. 0.2 corresponds to 80% train
 * and 20% test data)
 * @return train and test data sets
 **/
std::pair<MovieLens, MovieLens> readAndSplitMovieLens(const std::string& data_path, double test_size) {
    // read the data into a vector of MovieLensRatings
    // split into two vectors according to test_size
    // create Torch Datasets and return them
}

MovieLens::MovieLens(std::vector<MovieLensRating>& data, Mode mode = Mode::TRAIN) : m_mode(mode) {
    m_user_item_pairs = torch::empty({long(data.size()), 2}, torch::kInt32);
    m_ratings = torch::empty(data.size(), torch::kFloat);

    for (int i = 0; i < data.size(); i++) {
        m_user_item_pairs[i] = torch::tensor({data[i].itemID, data[i].userID}, torch::kInt32);
        m_ratings[i] = torch::tensor(data[i].rating, torch::kFloat);
    }
}

torch::data::Example<> MovieLens::get(size_t index) { return {m_user_item_pairs[index], m_ratings[index]}; }

torch::optional<size_t> MovieLens::size() const { return m_user_item_pairs.size(0); }

bool MovieLens::is_train() const noexcept { return m_mode == Mode::TRAIN; }

const torch::Tensor& MovieLens::getUserItemPairs() const { return m_user_item_pairs; }

const torch::Tensor& MovieLens::getRatings() const { return m_ratings; }