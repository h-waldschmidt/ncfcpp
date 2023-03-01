#include "dataset.h"

#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>

MovieLens::MovieLens(std::vector<MovieLensRating>& data, int64_t num_users, int64_t num_items, ProblemMode problem_mode,
                     Mode mode)
    : m_mode(mode), m_problem_mode(problem_mode), m_num_users(num_users), m_num_items(num_items) {
    m_user_item_pairs = torch::empty({long(data.size()), 2}, torch::kInt32);

    if (problem_mode == ProblemMode::CLASSIFICATION)
        m_ratings = torch::empty({long(data.size()), 5}, torch::kFloat);
    else
        m_ratings = torch::empty({long(data.size()), 1}, torch::kFloat);

    for (int i = 0; i < data.size(); i++) {
        m_user_item_pairs[i] = torch::tensor({data[i].userID, data[i].itemID}, torch::kInt32);
        m_ratings[i] = float(data[i].rating);

        if (problem_mode == ProblemMode::CLASSIFICATION) {
            m_ratings[i] = torch::zeros(5);
            m_ratings[i][data[i].rating - 1] = 1.0;
        } else {
            m_ratings[i] = torch::zeros(1);
            m_ratings[i][0] = data[i].rating;
        }
    }
}

torch::data::Example<> MovieLens::get(size_t index) { return {m_user_item_pairs[index], m_ratings[index]}; }

torch::optional<size_t> MovieLens::size() const { return m_user_item_pairs.size(0); }

bool MovieLens::is_train() const noexcept { return m_mode == Mode::TRAIN; }

const torch::Tensor& MovieLens::getUserItemPairs() const { return m_user_item_pairs; }

const torch::Tensor& MovieLens::getRatings() const { return m_ratings; }

const int64_t MovieLens::getNumOfUser() const { return m_num_users; }

const int64_t MovieLens::getNumOfItems() const { return m_num_items; }

const ProblemMode MovieLens::getProblemMode() const { return m_problem_mode; }

// split the ratings after each user
// this results in training and test data containing data for each user
void splitRatings(std::vector<MovieLensRating>& train_ratings, std::vector<MovieLensRating>& test_ratings,
                  double test_size, int num_of_users) {
    // sort the ratings after users
    std::vector<std::vector<MovieLensRating>> user_ratings(num_of_users);
    for (int i = 0; i < train_ratings.size(); i++) {
        MovieLensRating current_rating = train_ratings[i];
        user_ratings[current_rating.userID].push_back(current_rating);
    }

    // split ratings for each user into test and train
    train_ratings.clear();
    for (int i = 0; i < user_ratings.size(); i++) {
        int num_of_items = ceil(user_ratings[i].size() * test_size);

        // randomly select test data
        std::vector<bool> selected_for_test(user_ratings[i].size(), false);
        for (int j = 0; j < num_of_items; j++) {
            int random_rating_index = rand() % user_ratings[i].size();
            while (selected_for_test[random_rating_index]) {
                random_rating_index = rand() % num_of_items;
            }
            selected_for_test[random_rating_index] = true;
        }

        // push ratings into train and test vectors
        for (int j = 0; j < user_ratings[i].size(); j++) {
            if (selected_for_test[j]) {
                test_ratings.push_back(user_ratings[i][j]);
            } else {
                train_ratings.push_back(user_ratings[i][j]);
            }
        }
    }
}

std::pair<MovieLens, MovieLens> readAndSplitMovieLens(const std::string& data_path, double test_size,
                                                      ProblemMode problem_mode) {
    // read the data into a vector of MovieLensRatings
    std::vector<MovieLensRating> ratings;
    std::ifstream infile(data_path);
    std::string userId_string, itemId_string, rating_string;
    std::string line;
    int max_user = -1;
    int max_item = -1;
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
        if (current_rating.userID > max_user) {
            max_user = current_rating.userID;
        }
        if (current_rating.itemID > max_item) {
            max_item = current_rating.itemID;
        }
    }

    // split into two vectors according to test_size
    std::vector<MovieLensRating> test_ratings;
    splitRatings(ratings, test_ratings, test_size, max_user + 1);

    // create Torch Datasets and return them
    MovieLens train_data(ratings, max_user + 1, max_item + 1, problem_mode);
    MovieLens test_data(test_ratings, max_user + 1, max_item + 1, problem_mode, MovieLens::Mode::TEST);
    return std::make_pair(train_data, test_data);
}
