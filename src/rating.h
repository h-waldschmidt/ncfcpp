#pragma once

// defines a rating given by the MovieLens dataset
struct MovieLensRating {
    int userID;
    int itemID;
    double rating;
};

enum ProblemMode { REGRESSION, CLASSIFICATION };
