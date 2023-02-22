#include "dataset.h"

#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>

MovieLens::MovieLens(std::vector<MovieLensRating>& data, int64_t num_users, int64_t num_items, int64_t ouput_dims,
                     Mode mode)
    : m_mode(mode), m_num_users(num_users), m_num_items(num_items) {
    m_user_item_pairs = torch::empty({long(data.size()), 2}, torch::kInt32);
    m_ratings = torch::empty({long(data.size()), 5}, torch::kInt8);

    for (int i = 0; i < data.size(); i++) {
        m_user_item_pairs[i] = torch::tensor({data[i].itemID, data[i].userID}, torch::kInt32);
        m_ratings[i] = torch::zeros(5);
        m_ratings[i][data[i].rating - 1] = 1;
    }
}

torch::data::Example<> MovieLens::get(size_t index) { return {m_user_item_pairs[index], m_ratings[index]}; }

torch::optional<size_t> MovieLens::size() const { return m_user_item_pairs.size(0); }

bool MovieLens::is_train() const noexcept { return m_mode == Mode::TRAIN; }

const torch::Tensor& MovieLens::getUserItemPairs() const { return m_user_item_pairs; }

const torch::Tensor& MovieLens::getRatings() const { return m_ratings; }

void splitRatings(std::vector<MovieLensRating>& train_ratings, std::vector<MovieLensRating>& test_ratings,
                  double test_size) {
    int num_test_ratings = double(train_ratings.size()) * test_size;
    std::random_device dev;
    std::mt19937 rng(dev());

    for (int i = 0; i < num_test_ratings; i++) {
        std::uniform_int_distribution<std::mt19937::result_type> dist(0, train_ratings.size() - 1);
        int random_index = dist(rng);

        test_ratings.push_back(train_ratings[random_index]);
        train_ratings.erase(train_ratings.begin() + random_index);
    }
}

std::pair<MovieLens, MovieLens> readAndSplitMovieLens(const std::string& data_path, double test_size,
                                                      int64_t ouput_dims) {
    // read the data into a vector of MovieLensRatings
    std::vector<MovieLensRating> ratings;
    std::ifstream infile(data_path);
    std::string userId_string, itemId_string, rating_string;
    std::string line;
    std::unordered_set<int> users;
    std::unordered_set<int> items;

    while (getline(infile, line)) {
        std::stringstream ss(line);

        // file comes in format userID::itemID::rating::timestamp
        getline(ss, userId_string, ':');
        getline(ss, itemId_string, ':');
        getline(ss, itemId_string, ':');
        getline(ss, rating_string, ':');
        getline(ss, rating_string, ':');

        MovieLensRating current_rating = {stoi(userId_string), stoi(itemId_string), atof(rating_string.c_str())};
        ratings.push_back(current_rating);
        users.emplace(current_rating.userID);
        items.emplace(current_rating.itemID);
    }

    // split into two vectors according to test_size
    std::vector<MovieLensRating> test_ratings;
    splitRatings(ratings, test_ratings, test_size);

    // create Torch Datasets and return them
    MovieLens train_data(ratings, users.size(), items.size(), ouput_dims);
    MovieLens test_data(test_ratings, users.size(), items.size(), ouput_dims, MovieLens::Mode::TEST);
    return std::make_pair(train_data, test_data);
}