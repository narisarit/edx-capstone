# Rubrics 

# Files(10 points) 
# All 3 'Rmd report, PDF report, R script'

# Report(40 points)
# 'introduction/overview/executive summary' section that describes the data set and summarizes the goal of the project and key steps that were performed
# 'method/analysis' section that explains the process and techniques used, including data cleaning, data exploration and visualization, insights gained, and your modeling approach
# 'result' section that presents the modeling results and discusses the model performance
# 'conclusion' section that gives a brief summary of the report, its limitation and future work

# Code(25 points)
# well-commented and easy to follow

# RMSE(25 points)
# provide the appropriate score given the reported RMSE
# use final_holdout_test at this part
# 25 points: RMSE < 0.86490

#------------------------------------------------------------------

## Introduction / Overview / Executive summary

# This project uses a subset of the MovieLens 10M dataset provided in the `edx` object, which includes user ratings for movies along with metadata such as titles, genres, and timestamps. The goal is to build a recommendation system that minimizes the RMSE (Root Mean Squared Error) on a provided hold-out test set (`final_holdout_test`).

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# I need to get back to this section with key steps that were performed such as data cleaning, EDA, modeling, evaluation after I finish this project overall.


## Methods / Analysis
# Library libraries that we will use for this project
library(caret)
library(purrr)
library(tidyverse)
library(recosystem)
library(lubridate)
library(future.apply)
# After implementing the codes provided, we get two data objects, 'edx' and 'final_holdout_test'
# The latter is only for final assessment(RMSE calculation)
# Let's simply look at 'edx''s structure
head(edx) # show first few rows
str(edx) # show structure of edx
all(complete.cases(edx)) # check if there's any NAs
# Since there were no NAs We could skip dropping NAs

# Mutate timestamp to YYYY-MM-DD format.
edx <- edx %>%
  mutate(rating_date = as.POSIXct(timestamp, origin = "1970-01-01"))

# edx has 6 variables and 9000055 observations. What we want to predict is the variable 'rating'

# Since we have to use only 'edx' object, we split 'edx' into train_set and test_set
# Function to split each user's data randomly
split_by_user <- function(user_df, p = 0.8) {
  n <- nrow(user_df)
  if (n == 1) {
    # keep single ratings in train to avoid dropping users
    list(train = user_df, test = NULL)
  } else {
    idx <- createDataPartition(1:n, p = p, list = FALSE)
    list(
      train = user_df[idx, ],
      test  = user_df[-idx, ]
    )
  }
}

# Apply the function per user
splits <- edx %>%
  group_by(userId) %>%
  group_split() %>%
  map(split_by_user)

# Combine into full train and test sets
edx_train <- map_dfr(splits, "train")
edx_test  <- map_dfr(splits, "test") %>%
  filter(!is.null(.))  # remove empty cases (if any)

# Optional: ensure no cold-start problems
edx_test <- edx_test %>%
  filter(movieId %in% edx_train$movieId,
         userId %in% edx_train$userId)

# save objects locally(NOT for SUbMISSION)
getwd()
list.files()
dir.exists("data")
list.files("data")
file.path("data")
saveRDS(edx, file = "data/edx.rds")
saveRDS(edx_train, file = "data/edx_train.rds")
saveRDS(edx_test, file = "data/edx_test.rds")
edx <- readRDS("data/edx.rds")
edx_train <- readRDS("data/edx_train.rds")
edx_test <- readRDS("data/edx_test.rds")

# Now let's do Exploratory Data Analysis
str(edx)
summary(edx)
edx %>% summarize(
  num_users = n_distinct(userId),
  num_movies = n_distinct(movieId)
)
# there are 69878 users, 10677 movies

# let's see rating's distribution
edx %>%
  ggplot(aes(x = rating)) +
  geom_bar() +
  labs(title = "Distribution of Movie Ratings", x = "Rating", y = "Count")
# we could see the distribution of rating's is skewed to the left. And whole numbers are more frequent than .5s

# let's see ratings per user
edx %>%
  count(userId) %>%
  ggplot(aes(x = n)) +
  geom_histogram(bins = 50) +
  scale_x_log10() +
  labs(title = "Number of Ratings per User", x = "Ratings per User (log scale)", y = "Number of Users")
# ratings per user is skewed to the right. Most users rated a few times but some users rated very many times.


# let' see ratings per movie
edx %>%
  count(movieId) %>%
  ggplot(aes(x = n)) +
  geom_histogram(bins = 50) +
  scale_x_log10() +
  labs(title = "Number of Ratings per Movie", x = "Ratings per Movie (log scale)", y = "Number of Movies")
# ratings per movie is skewed to the right. Most movies were rated a few times but some movies were rated very many times.

# let's see ratings per genres.
edx %>%
  separate_rows(genres, sep = "\\|") %>%
  count(genres, sort = TRUE) %>%
  top_n(15) %>%
  ggplot(aes(x = reorder(genres, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top 15 Most Common Genres", x = "Genre", y = "Number of Ratings")
# Genres drama, comedy, action, thriller were rated a lot. 

# let's see ratings over time
edx %>%
  ggplot(aes(x = rating_date)) +
  geom_histogram(bins = 30) +
  labs(title = "Rating Frequency Over Time", x = "Date", y = "Number of Ratings")
# ratings changed over time but irregularly.

# Now we've finished distributions of variables, we're gonna look at correlations between rating and other variables.

# plot rating_year and rating
edx_year <- edx %>%
  mutate(rating_date = as.POSIXct(timestamp, origin = "1970-01-01"),
         rating_year = lubridate::year(rating_date))

ggplot(edx_year, aes(x = rating_year, y = rating)) +
  geom_jitter(alpha = 0.1) +
  geom_smooth(method = "loess") +
  labs(title = "Rating vs Year", x = "Year", y = "Rating")

# Correlation (numeric)
cor(as.numeric(edx_year$timestamp), edx_year$rating)
# We could see there is no significant correlation between rating_year and rating. So we could delay including timestamp for our model, when there is enough accuracy without it.

# Correlation between rating and user
# get average by userId
user_avg <- edx %>%
  group_by(userId) %>%
  summarize(user_mean = mean(rating), user_count = n())

# plot user_mean and user_count
user_avg %>% 
  ggplot(aes(x = user_mean, y = user_count)) +
  geom_point()
# We could see that some people rate higher than others and some lower. Except a few outliers this distribution seems well centered.

# Join to edx
edx_user <- edx %>%
  left_join(user_avg, by = "userId")

# Correlation between user's average rating and their individual ratings
cor(edx_user$user_mean, edx_user$rating)
# We could see there is a significant positive correlation between user_mean and their rating. It sounds natural. We should include user effect to our algorithm.

# Correlation between rating and movie
movie_avg <- edx %>%
  group_by(movieId) %>%
  summarize(movie_mean = mean(rating), movie_count = n())

#plot movie mean and movie count
movie_avg %>%
  ggplot(aes(x = movie_mean, y = movie_count)) +
  geom_point()
# We could see that highly rated movies were much frequently rated. It sounds natural, because people will watch good movie more times.

edx_movie <- edx %>%
  left_join(movie_avg, by = "movieId")

cor(edx_movie$movie_mean, edx_movie$rating)
# We could see that there is a significant positive correlation between movie mean and movie rating. It sounds natural. We should include movie effect to our algorithm.

# Correlation between rating and movie release year
edx_release <- edx %>%
  mutate(release_year = str_extract(title, "\\(\\d{4}\\)")) %>%
  mutate(release_year = as.numeric(str_remove_all(release_year, "[()]")))

head(edx_release)

cor(edx_release$release_year, edx_release$rating, use = "complete.obs") 

ggplot(edx_release, aes(x = release_year, y = rating)) +
  geom_point(alpha = 0.1) +
  geom_smooth(method = "loess") +
  labs(title = "Rating vs Release Year")
# We could see there is weak negative correlation between release year and rating. But when we see trend line over time using Loess smoothing, we could see a transition to positive correlation after late 1990s. May be we could include this non-linear relationship to our algorithm.

# Correlation between genres and ratings
edx %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(mean_rating = mean(rating), n = n()) %>%
  arrange(desc(mean_rating)) %>%
  ggplot(aes(x = reorder(genres, mean_rating), y = mean_rating)) +
  geom_col() +
  coord_flip() +
  labs(title = "Average Rating by Genre", x = "Genre", y = "Rating")
# We could see that some genres are rated higher than others. May be we could include genres to our algorithm.

### Key Insights from Exploratory Data Analysis

# The dataset contains **69,878 unique users** and **10,677 unique movies**.
# Ratings are mostly whole numbers between 1 and 5, with a strong left skew (more high ratings).
# Most users rated only a few movies, but some were extremely active.
# Most movies received only a few ratings, while some blockbusters were rated frequently.
# The most common genres are **Drama**, **Comedy**, **Action**, and **Thriller**.
# There’s no strong correlation between **rating timestamp** and the rating itself.
# Some users consistently rate higher or lower than others → **user effect** exists.
# Some movies are consistently rated higher/lower → **movie effect** exists.
# Movies released after the late 1990s tend to show slightly increasing rating trends over time.
# Some genres have higher average ratings than others

# Modeling approach
# Baseline model: predict global average
mu <- mean(edx_train$rating)
rmse_baseline <- RMSE(edx_test$rating, mu)
print(paste("Baseline RMSE (global mean only):", round(rmse_baseline, 5)))
# we have long way to go lower than 0.86490

# Movie Effect Model
movie_avgs <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

predicted_ratings <- edx_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i)

rmse_movie <- RMSE(predicted_ratings$rating, predicted_ratings$pred)
print(paste("Movie Effect RMSE:", round(rmse_movie, 5)))
# we much improved than naive model.

# Movie + User Effect Model
# User effect (b_u), adjusted for movie bias
user_avgs <- edx_train %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict ratings on edx_test
predicted_ratings <- edx_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u)

# Calculate RMSE
rmse_user_movie <- RMSE(predicted_ratings$rating, predicted_ratings$pred)
print(paste("Movie + User Effect RMSE:", round(rmse_user_movie, 5)))
# We much improved than movie effect only model.

# Regularized model to prevent over-fitting on rare movies/users
# Regularized Movie + User Effect Model
plan(multisession, workers = parallel::detectCores() - 1) 
options(future.globals.maxSize = 2 * 1024^3)

lambdas <- seq(0, 10, 0.25)  # Try multiple lambda values

rmse_results <- future_sapply(lambdas, function(lambda) {
  mu <- mean(edx_train$rating)
  
  # Regularized movie effect
  b_i <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  # Regularized user effect
  b_u <- edx_train %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))
  
  # Predictions
  predicted_ratings <- edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(edx_test$rating, predicted_ratings))
})
plan(sequential)

# Find best lambda
best_lambda <- lambdas[which.min(rmse_results)]
rmse_regularized_movie_user <- min(rmse_results)

print(paste("Best lambda:", best_lambda))
print(paste("Regularized RMSE:", round(rmse_regularized_movie_user, 5)))
# We decreased RMSE a little bit.


# Let's try including timestamp to our model
# Ensure date/year columns exist
plan(multisession, workers = parallel::detectCores() - 1) 

edx_train <- edx_train %>%
  mutate(rating_year = year(as_datetime(timestamp)))

edx_test <- edx_test %>%
  mutate(rating_year = year(as_datetime(timestamp)))

# Set lambdas to try
lambdas <- seq(0, 10, 0.25)

# RMSE loop
rmse_results <- future_sapply(lambdas, function(lambda) {
  mu <- mean(edx_train$rating)
  
  # Regularized movie effect
  b_i <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  # Regularized user effect
  b_u <- edx_train %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))
  
  # Regularized time (year) effect
  b_t <- edx_train %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(rating_year) %>%
    summarize(b_t = sum(rating - mu - b_i - b_u) / (n() + lambda))
  
  # Predict on edx_test
  predicted <- edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = "rating_year") %>%
    mutate(pred = mu + b_i + b_u + b_t) %>%
    pull(pred)
  
  return(RMSE(edx_test$rating, predicted))
})
plan(sequential)

# Best lambda
best_lambda <- lambdas[which.min(rmse_results)]
rmse_regularized_movie_user_timestamp <- min(rmse_results)

print(paste("Best lambda:", best_lambda))
print(paste("Regularized Movie + User + Year RMSE:", round(rmse_regularized_movie_user_timestamp, 5)))
# Including timestamp to algorithm decreased RMSE on testset by only 0.00001. We think this might lead to overfitting and too heavy processing so we will exclude timestamp. best_lambda was same between the two models that included/excluded timestamp, so there is no need to re-assgin best lambda.

# Let's do matrix factorization on residuals using 'recosystem' library
lambda <- best_lambda  
mu <- mean(edx_train$rating)

# Regularized movie effect
b_i <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda))

# Regularized user effect
b_u <- edx_train %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))

# Compute residuals
edx_train_resid <- edx_train %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(residual = rating - mu - b_i - b_u)

# Create MF training data using residuals
train_data <- data_memory(user_index = edx_train_resid$userId,
                          item_index = edx_train_resid$movieId,
                          rating = edx_train_resid$residual)

r_hybrid <- Reco()

# Tune
opts <- r_hybrid$tune(train_data, opts = list(dim = c(10, 20, 30), 
                                       costp_l2 = c(0.01, 0.1),
                                       costq_l2 = c(0.01, 0.1),
                                       niter = 10))

# saving object is NOT for submission
saveRDS(opts, file = "data/opts.rds")
opts <- readRDS(file = "data/opts.rds")

# Train on residuals
r_hybrid$train(train_data, opts = c(opts$min, niter = 20))

# predict MF residuals on edx_test
edx_test_mf <- edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId")

# Prepare test data for MF
test_data <- data_memory(user_index = edx_test_mf$userId,
                         item_index = edx_test_mf$movieId)

# Predict residuals
pred_resid <- r_hybrid$predict(test_data, out_memory())

# combine predictions
edx_test_mf <- edx_test_mf %>%
  mutate(pred_manual = mu + b_i + b_u,
         pred_final = pred_manual + pred_resid)

# Calculate RMSE
rmse_hybrid <- RMSE(edx_test_mf$rating, edx_test_mf$pred_final)
print(paste("Hybrid (Manual + MF residual) RMSE:", round(rmse_hybrid, 5)))
# We got much lower RMSE (lower that 0.8)

# The models we made is 'manual model' and 'manual + matrix factorization model'
# Let's try 'matrix factorization only model'
# Make sure userId and movieId are integers and matched
train_data <- data_memory(
  user_index = edx_train$userId,
  item_index = edx_train$movieId,
  rating = edx_train$rating
)

test_data <- data_memory(
  user_index = edx_test$userId,
  item_index = edx_test$movieId
)

r_mf_only <- Reco()

opts_mf_only <- r_mf_only$tune(train_data, opts = list(
  dim = c(10, 20, 30),
  costp_l2 = c(0.01, 0.1),
  costq_l2 = c(0.01, 0.1),
  nthread = 2,
  niter = 10
))

# saving object is NOT for submission
saveRDS(opts_mf_only, file = "data/opts_mf_only.rds")
opts_mf_only <- readRDS(file = "data/opts_mf_only.rds")

r_mf_only$train(train_data, opts = c(opts_mf_only$min, niter = 20))

pred <- r_mf_only$predict(test_data, out_memory())

# Evaluate RMSE
rmse_mf_only <- RMSE(edx_test$rating, pred)
print(paste("Pure Matrix Factorization RMSE:", round(rmse_mf_only, 5)))
# Surprisingly and also sadly pure matrix factorization model's RMSE was the lowest.

# Results
# Create RMSE comparison data frame
rmse_results_df <- data.frame(
  Model = c(
    "Global Mean Only",
    "Movie Effect",
    "Movie + User Effect",
    "Regularized Movie + User",
    "Manual + MF Residual (Hybrid)",
    "Pure Matrix Factorization"
  ),
  RMSE = c(
    rmse_baseline,
    rmse_movie,
    rmse_user_movie,
    rmse_regularized_movie_user,
    rmse_hybrid,
    rmse_mf_only
  )
)

print(rmse_results_df)

ggplot(rmse_results_df, aes(x = reorder(Model, RMSE), y = RMSE)) +
  geom_col(fill = "steelblue") +
  geom_text(aes(label = round(RMSE, 4)), vjust = -0.3, size = 3.5) +
  labs(title = "RMSE Comparison by Model",
       x = "Model Type", y = "RMSE") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# Now let's test on final_holdout_test
# Prepare final_holdout_test for hybrid prediction
final_holdout_test_hybrid <- final_holdout_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(b_i = ifelse(is.na(b_i), 0, b_i),
         b_u = ifelse(is.na(b_u), 0, b_u))

# Create data for MF residual prediction
final_mf_data <- data_memory(
  user_index = final_holdout_test_hybrid$userId,
  item_index = final_holdout_test_hybrid$movieId
)

# Predict MF residuals
pred_resid_final <- r_hybrid$predict(final_mf_data, out_memory())

# Combine manual and MF residual
final_holdout_test_hybrid <- final_holdout_test_hybrid %>%
  mutate(pred_manual = mu + b_i + b_u,
         pred_final = pred_manual + pred_resid_final)

# Compute RMSE
rmse_hybrid_final <- RMSE(final_holdout_test_hybrid$rating, final_holdout_test_hybrid$pred_final)
print(paste("Hybrid Model RMSE on final_holdout_test:", round(rmse_hybrid_final, 5)))


# retrain Matrix factorization only model on full edx
# Step 1: Create training set using full edx
train_data_edx <- data_memory(
  user_index = edx$userId,
  item_index = edx$movieId,
  rating = edx$rating
)

# Step 2: Initialize and tune
r_final <- Reco()

opts_final <- r_final$tune(train_data_edx, opts = list(
  dim = c(10, 20, 30),
  costp_l2 = c(0.01, 0.1),
  costq_l2 = c(0.01, 0.1),
  niter = 10,
  nthread = 2
))

# Optional: save tuning results
saveRDS(opts_final, file = "data/opts_final_edx.rds")

# Step 3: Train using best options
r_final$train(train_data_edx, opts = c(opts_final$min, niter = 20))

# Prepare final_holdout_test for prediction
final_data <- data_memory(
  user_index = final_holdout_test$userId,
  item_index = final_holdout_test$movieId
)

# Predict
pred_final_mf <- r_final$predict(final_data, out_memory())

# Evaluate RMSE
rmse_final_mf <- RMSE(final_holdout_test$rating, pred_final_mf)
print(paste("Pure Matrix Factorization RMSE on final_holdout_test:", round(rmse_final_mf, 5)))

