import os
from databricks_test_helper import Test

dbfs_dir = './data-001/'
ratings_filename = dbfs_dir + '/ratings.csv'
movies_filename = dbfs_dir + '/movies.csv'

# The following line is here to enable this notebook to be exported as source and
# run on a local machine with a local copy of the files. Just change the dbfs_dir,
# above.
if os.path.sep != '/':
  # Handle Windows.
  ratings_filename = ratings_filename.replace('/', os.path.sep)
  movie_filename = movie_filename.replace('/', os.path.sep)

from pyspark.sql.types import *

ratings_df_schema = StructType(
  [StructField('userId', IntegerType()),
   StructField('movieId', IntegerType()),
   StructField('rating', DoubleType())]
)
movies_df_schema = StructType(
  [StructField('ID', IntegerType()),
   StructField('title', StringType())]
)

from pyspark.sql.functions import regexp_extract
from pyspark.sql.types import *

raw_ratings_df = sqlContext.read.format('com.databricks.spark.csv').options(header=True, inferSchema=False).schema(ratings_df_schema).load(ratings_filename)
ratings_df = raw_ratings_df.drop('Timestamp')

raw_movies_df = sqlContext.read.format('com.databricks.spark.csv').options(header=True, inferSchema=False).schema(movies_df_schema).load(movies_filename)
movies_df = raw_movies_df.drop('Genres').withColumnRenamed('movieId', 'ID')

ratings_df.cache()
movies_df.cache()

assert ratings_df.is_cached
assert movies_df.is_cached

raw_ratings_count = raw_ratings_df.count()
ratings_count = ratings_df.count()
raw_movies_count = raw_movies_df.count()
movies_count = movies_df.count()

print 'There are %s ratings and %s movies in the datasets' % (ratings_count, movies_count)
print 'Ratings:'
ratings_df.show(3)
print 'Movies:'
movies_df.show(3, truncate=False)

assert raw_ratings_count == ratings_count
assert raw_movies_count == movies_count

assert ratings_count == 20000263
assert movies_count == 27278
assert movies_df.filter(movies_df.title == 'Toy Story (1995)').count() == 1
assert ratings_df.filter((ratings_df.userId == 6) & (ratings_df.movieId == 1) & (ratings_df.rating == 5.0)).count() == 1

display(movies_df)
display(ratings_df)

# TODO: Replace <FILL_IN> with appropriate code
from pyspark.sql import functions as F

# From ratingsDF, create a movie_ids_with_avg_ratings_df that combines the two DataFrames
movie_ids_with_avg_ratings_df = ratings_df.groupBy('movieId').agg(F.count(ratings_df.rating).alias("count"), F.avg(ratings_df.rating).alias("average"))
print 'movie_ids_with_avg_ratings_df:'
movie_ids_with_avg_ratings_df.show(3, truncate=False)

# Note: movie_names_df is a temporary variable, used only to separate the steps necessary
# to create the movie_names_with_avg_ratings_df DataFrame.
movie_names_df = movie_ids_with_avg_ratings_df.<FILL_IN>
movie_names_with_avg_ratings_df = movie_names_df.<FILL_IN>

print 'movie_names_with_avg_ratings_df:'
movie_names_with_avg_ratings_df.show(3, truncate=False)

# TEST Movies with Highest Average Ratings (1a)
Test.assertEquals(movie_ids_with_avg_ratings_df.count(), 26744,
                'incorrect movie_ids_with_avg_ratings_df.count() (expected 26744)')
movie_ids_with_ratings_take_ordered = movie_ids_with_avg_ratings_df.orderBy('MovieID').take(3)
_take_0 = movie_ids_with_ratings_take_ordered[0]
_take_1 = movie_ids_with_ratings_take_ordered[1]
_take_2 = movie_ids_with_ratings_take_ordered[2]
Test.assertTrue(_take_0[0] == 1 and _take_0[1] == 49695,
                'incorrect count of ratings for movie with ID {0} (expected 49695)'.format(_take_0[0]))
Test.assertEquals(round(_take_0[2], 2), 3.92, "Incorrect average for movie ID {0}. Expected 3.92".format(_take_0[0]))

Test.assertTrue(_take_1[0] == 2 and _take_1[1] == 22243,
                'incorrect count of ratings for movie with ID {0} (expected 22243)'.format(_take_1[0]))
Test.assertEquals(round(_take_1[2], 2), 3.21, "Incorrect average for movie ID {0}. Expected 3.21".format(_take_1[0]))

Test.assertTrue(_take_2[0] == 3 and _take_2[1] == 12735,
                'incorrect count of ratings for movie with ID {0} (expected 12735)'.format(_take_2[0]))
Test.assertEquals(round(_take_2[2], 2), 3.15, "Incorrect average for movie ID {0}. Expected 3.15".format(_take_2[0]))


Test.assertEquals(movie_names_with_avg_ratings_df.count(), 26744,
                  'incorrect movie_names_with_avg_ratings_df.count() (expected 26744)')
movie_names_with_ratings_take_ordered = movie_names_with_avg_ratings_df.orderBy(['average', 'title']).take(3)
result = [(r['average'], r['title'], r['count'], r['movieId']) for r in movie_names_with_ratings_take_ordered]
Test.assertEquals(result,
                  [(0.5, u'13 Fighting Men (1960)', 1, 109355),
                   (0.5, u'20 Years After (2008)', 1, 131062),
                   (0.5, u'3 Holiday Tails (Golden Christmas 2: The Second Tail, A) (2011)', 1, 111040)],
                  'incorrect top 3 entries in movie_names_with_avg_ratings_df')

# TODO: Replace <FILL IN> with appropriate code
movies_with_500_ratings_or_more = movie_names_with_avg_ratings_df.<FILL_IN>
print 'Movies with highest ratings:'
movies_with_500_ratings_or_more.show(20, truncate=False)

# TEST Movies with Highest Average Ratings and at least 500 Reviews (1b)

Test.assertEquals(movies_with_500_ratings_or_more.count(), 4489,
                  'incorrect movies_with_500_ratings_or_more.count(). Expected 4489.')
top_20_results = [(r['average'], r['title'], r['count']) for r in movies_with_500_ratings_or_more.orderBy(F.desc('average')).take(20)]

Test.assertEquals(top_20_results,
                  [(4.446990499637029, u'Shawshank Redemption, The (1994)', 63366),
                   (4.364732196832306, u'Godfather, The (1972)', 41355),
                   (4.334372207803259, u'Usual Suspects, The (1995)', 47006),
                   (4.310175010988133, u"Schindler's List (1993)", 50054),
                   (4.275640557704942, u'Godfather: Part II, The (1974)', 27398),
                   (4.2741796572216, u'Seven Samurai (Shichinin no samurai) (1954)', 11611),
                   (4.271333600779414, u'Rear Window (1954)', 17449),
                   (4.263182346109176, u'Band of Brothers (2001)', 4305),
                   (4.258326830670664, u'Casablanca (1942)', 24349),
                   (4.256934865900383, u'Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)', 6525),
                   (4.24807897901911, u"One Flew Over the Cuckoo's Nest (1975)", 29932),
                   (4.247286821705426, u'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)', 23220),
                   (4.246001523229246, u'Third Man, The (1949)', 6565),
                   (4.235410064157069, u'City of God (Cidade de Deus) (2002)', 12937),
                   (4.2347902097902095, u'Lives of Others, The (Das leben der Anderen) (2006)', 5720),
                   (4.233538107122288, u'North by Northwest (1959)', 15627),
                   (4.2326233183856505, u'Paths of Glory (1957)', 3568),
                   (4.227123123722136, u'Fight Club (1999)', 40106),
                   (4.224281931146873, u'Double Indemnity (1944)', 4909),
                   (4.224137931034483, u'12 Angry Men (1957)', 12934)],
                  'Incorrect top 20 movies with 500 or more ratings')

# TODO: Replace <FILL_IN> with the appropriate code.

# We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing
seed = 1800009193L
(split_60_df, split_a_20_df, split_b_20_df) = <FILL_IN>

# Let's cache these datasets for performance
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()

print('Training: {0}, validation: {1}, test: {2}\n'.format(
  training_df.count(), validation_df.count(), test_df.count())
)
training_df.show(3)
validation_df.show(3)
test_df.show(3)

# TEST Creating a Training Set (2a)
Test.assertEquals(training_df.count(), 12001389, "Incorrect training_df count. Expected 12001389")
Test.assertEquals(validation_df.count(), 4003694, "Incorrect validation_df count. Expected 4003694")
Test.assertEquals(test_df.count(), 3995180, "Incorrect test_df count. Expected 3995180")

Test.assertEquals(training_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 5952) & (ratings_df.rating == 5.0)).count(), 1)
Test.assertEquals(training_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 1193) & (ratings_df.rating == 3.5)).count(), 1)
Test.assertEquals(training_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 1196) & (ratings_df.rating == 4.5)).count(), 1)

Test.assertEquals(validation_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 296) & (ratings_df.rating == 4.0)).count(), 1)
Test.assertEquals(validation_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 32) & (ratings_df.rating == 3.5)).count(), 1)
Test.assertEquals(validation_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 6888) & (ratings_df.rating == 3.0)).count(), 1)

Test.assertEquals(test_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 4993) & (ratings_df.rating == 5.0)).count(), 1)
Test.assertEquals(test_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 4128) & (ratings_df.rating == 4.0)).count(), 1)
Test.assertEquals(test_df.filter((ratings_df.userId == 1) & (ratings_df.movieId == 4915) & (ratings_df.rating == 3.0)).count(), 1)

# TODO: Replace <FILL IN> with appropriate code
# This step is broken in ML Pipelines: https://issues.apache.org/jira/browse/SPARK-14489
from pyspark.ml.recommendation import ALS

# Let's initialize our ALS learner
als = ALS()

# Now we set the parameters for the method
als.setMaxIter(5)\
   .setSeed(seed)\
   .setRegParam(0.1)\
   .<FILL_IN>

# Now let's compute an evaluation metric for our test dataset
from pyspark.ml.evaluation import RegressionEvaluator

# Create an RMSE evaluator using the label and predicted columns
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse")

tolerance = 0.03
ranks = [4, 8, 12]
errors = [0, 0, 0]
models = [0, 0, 0]
err = 0
min_error = float('inf')
best_rank = -1
for rank in ranks:
  # Set the rank here:
  als.<FILL_IN>
  # Create the model with these parameters.
  model = als.fit(training_df)
  # Run the model to create a prediction. Predict against the validation_df.
  predict_df = model.<FILL_IN>

  # Remove NaN values from prediction (due to SPARK-14489)
  predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))

  # Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
  error = reg_eval.<FILL_IN>
  errors[err] = error
  models[err] = model
  print 'For rank %s the RMSE is %s' % (rank, error)
  if error < min_error:
    min_error = error
    best_rank = err
  err += 1

als.setRank(ranks[best_rank])
print 'The best model was trained with rank %s' % ranks[best_rank]
my_model = models[best_rank]

# TEST
Test.assertEquals(round(min_error, 2), 0.81, "Unexpected value for best RMSE. Expected rounded value to be 0.81. Got {0}".format(round(min_error, 2)))
Test.assertEquals(ranks[best_rank], 12, "Unexpected value for best rank. Expected 12. Got {0}".format(ranks[best_rank]))
Test.assertEqualsHashed(als.getItemCol(), "18f0e2357f8829fe809b2d95bc1753000dd925a6", "Incorrect choice of {0} for ALS item column.".format(als.getItemCol()))
Test.assertEqualsHashed(als.getUserCol(), "db36668fa9a19fde5c9676518f9e86c17cabf65a", "Incorrect choice of {0} for ALS user column.".format(als.getUserCol()))
Test.assertEqualsHashed(als.getRatingCol(), "3c2d687ef032e625aa4a2b1cfca9751d2080322c", "Incorrect choice of {0} for ALS rating column.".format(als.getRatingCol()))

# TODO: Replace <FILL_IN> with the appropriate code
# In ML Pipelines, this next step has a bug that produces unwanted NaN values. We
# have to filter them out. See https://issues.apache.org/jira/browse/SPARK-14489
predict_df = my_model.<FILL_IN>

# Remove NaN values from prediction (due to SPARK-14489)
predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))

# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_df DataFrame
test_RMSE = <FILL_IN>

print('The model had a RMSE on the test set of {0}'.format(test_RMSE))

# TEST Testing Your Model (2c)
Test.assertTrue(abs(test_RMSE - 0.809624038485) < tolerance, 'incorrect test_RMSE: {0:.11f}'.format(test_RMSE))

# TODO: Replace <FILL_IN> with the appropriate code.
# Compute the average rating
avg_rating_df = <FILL_IN>

# Extract the average rating value. (This is row 0, column 0.)
training_avg_rating = avg_rating_df.collect()[0][0]

print('The average rating for movies in the training set is {0}'.format(training_avg_rating))

# Add a column with the average rating
test_for_avg_df = test_df.withColumn('prediction', <FILL_IN>)

# Run the previously created RMSE evaluator, reg_eval, on the test_for_avg_df DataFrame
test_avg_RMSE = <FILL_IN>

print("The RMSE on the average set is {0}".format(test_avg_RMSE))

# TEST Comparing Your Model (2d)
Test.assertTrue(abs(training_avg_rating - 3.52547984237) < 0.000001,
                'incorrect training_avg_rating (expected 3.52547984237): {0:.11f}'.format(training_avg_rating))
Test.assertTrue(abs(test_avg_RMSE - 1.05190953037) < 0.000001,
                'incorrect test_avg_RMSE (expected 1.0519743756): {0:.11f}'.format(test_avg_RMSE))

print 'Most rated movies:'
print '(average rating, movie name, number of reviews, movie ID)'
display(movies_with_500_ratings_or_more.orderBy(movies_with_500_ratings_or_more['average'].desc()).take(50))

# TODO: Replace <FILL IN> with appropriate code
from pyspark.sql import Row
my_user_id = 0

# Note that the movie IDs are the *last* number on each line. A common error was to use the number of ratings as the movie ID.
my_rated_movies = [
     <FILL IN>
     # The format of each line is (my_user_id, movie ID, your rating)
     # For example, to give the movie "Star Wars: Episode IV - A New Hope (1977)" a five rating, you would add the following line:
     #   (my_user_id, 260, 5),
]

my_ratings_df = sqlContext.createDataFrame(my_rated_movies, ['userId','movieId','rating'])
print 'My movie ratings:'
display(my_ratings_df.limit(10))


# TODO: Replace <FILL IN> with appropriate code
training_with_my_ratings_df = <FILL IN>

print ('The training dataset now has %s more entries than the original training dataset' %
       (training_with_my_ratings_df.count() - training_df.count()))
assert (training_with_my_ratings_df.count() - training_df.count()) == my_ratings_df.count()

# TODO: Replace <FILL IN> with appropriate code

# Reset the parameters for the ALS object.
als.setPredictionCol("prediction")\
   .setMaxIter(5)\
   .setSeed(seed)\
   .<FILL_IN>

# Create the model with these parameters.
my_ratings_model = als.<FILL_IN>

# TODO: Replace <FILL IN> with appropriate code
my_predict_df = my_ratings_model.<FILL IN>

# Remove NaN values from prediction (due to SPARK-14489)
predicted_test_my_ratings_df = my_predict_df.filter(my_predict_df.prediction != float('nan'))

# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_my_ratings_df DataFrame
test_RMSE_my_ratings = <FILL IN>
print('The model had a RMSE on the test set of {0}'.format(test_RMSE_my_ratings))

# TODO: Replace <FILL_IN> with the appropriate code

# Create a list of my rated movie IDs
my_rated_movie_ids = [x[1] for x in my_rated_movies]

# Filter out the movies I already rated.
not_rated_df = movies_df.<FILL_IN>

# Rename the "ID" column to be "movieId", and add a column with my_user_id as "userId".
my_unrated_movies_df = not_rated_df.<FILL_IN>

# Use my_rating_model to predict ratings for the movies that I did not manually rate.
raw_predicted_ratings_df = my_ratings_model.<FILL_IN>

predicted_ratings_df = raw_predicted_ratings_df.filter(raw_predicted_ratings_df['prediction'] != float('nan'))

# TODO: Replace <FILL_IN> with the appropriate code

predicted_with_counts_df = <FILL_IN>
predicted_highest_rated_movies_df = predicted_with_counts_df.<FILL_IN>

print ('My 25 highest rated movies as predicted (for movies with more than 75 reviews):')
predicted_highest_rated_movies_df.<FILL_IN>

