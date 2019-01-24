# labVersion = 'PPMLPA-1.0.0'


powerPlantDF = sqlContext.read.format(<FILL_IN>).options(<FILL_IN>).load(<FILL_IN>)

# TEST
from databricks_test_helper import *
expected = set([(s, 'double') for s in ('AP', 'AT', 'PE', 'RH', 'V')])
Test.assertEquals(expected, set(powerPlantDF.dtypes), "Incorrect schema for powerPlantDF")


print powerPlantDF.dtypes

display(powerPlantDF)

# TO DO: Fill in the custom schema.
from pyspark.sql.types import *

# Custom Schema for Power Plant
customSchema = StructType([ \
    <FILL_IN>, \
    <FILL_IN>, \
    <FILL_IN>, \
    <FILL_IN>, \
    <FILL_IN> \
                          ])

# TEST
Test.assertEquals(set([f.name for f in customSchema.fields]), set(['AT', 'V', 'AP', 'RH', 'PE']), 'Incorrect column names in schema.')
Test.assertEquals(set([f.dataType for f in customSchema.fields]), set([DoubleType(), DoubleType(), DoubleType(), DoubleType(), DoubleType()]), 'Incorrect column types in schema.')

# TODO: Use the schema you created above to load the data again.
altPowerPlantDF = sqlContext.read.format(<FILL_IN>).options(<FILL_IN>).load(<FILL_IN>)

# TEST
from databricks_test_helper import *
expected = set([(s, 'double') for s in ('AP', 'AT', 'PE', 'RH', 'V')])
Test.assertEquals(expected, set(altPowerPlantDF.dtypes), "Incorrect schema for powerPlantDF")

sqlContext.sql("DROP TABLE IF EXISTS power_plant")
dbutils.fs.rm("dbfs:/user/hive/warehouse/power_plant", True)
sqlContext.registerDataFrameAsTable(powerPlantDF, "power_plant")

%sql
-- We can use %sql to query the rows
SELECT * FROM power_plant

%sql
desc power_plant

df = sqlContext.table("power_plant")
display(df.describe())

%sql
select AT as Temperature, PE as Power from power_plant

%sql
-- TO DO: Replace <FILL_IN> with the appropriate SQL command.

% sql
-- TO
DO: Replace < FILL_IN >
with the appropriate SQL command.
< FILL_IN >

%sql
-- TO DO: Replace <FILL_IN> with the appropriate SQL command.
<FILL_IN>

# TODO: Replace <FILL_IN> with the appropriate code
from pyspark.ml.feature import VectorAssembler

datasetDF = <FILL_IN>

vectorizer = VectorAssembler()
vectorizer.setInputCols(<FILL_IN>)
vectorizer.setOutputCol(<FILL_IN>)

# TEST
Test.assertEquals(set(vectorizer.getInputCols()), {"AT", "V", "AP", "RH"}, "Incorrect vectorizer input columns")
Test.assertEquals(vectorizer.getOutputCol(), "features", "Incorrect vectorizer output column")

# TODO: Replace <FILL_IN> with the appropriate code.
# We'll hold out 20% of our data for testing and leave 80% for training
seed = 1800009193L
(split20DF, split80DF) = datasetDF.<FILL_IN>

# Let's cache these datasets for performance
testSetDF = <FILL_IN>
trainingSetDF = <FILL_IN>

# TEST
Test.assertEquals(trainingSetDF.count(), 38243, "Incorrect size for training data set")
Test.assertEquals(testSetDF.count(), 9597, "Incorrect size for test data set")

# ***** LINEAR REGRESSION MODEL ****

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml import Pipeline

# Let's initialize our linear regression learner
lr = LinearRegression()

# We use explain params to dump the parameters we can use
print(lr.explainParams())

# Now we set the parameters for the method
lr.setPredictionCol("Predicted_PE")\
  .setLabelCol("PE")\
  .setMaxIter(100)\
  .setRegParam(0.1)


# We will use the new spark.ml pipeline API. If you have worked with scikit-learn this will be very familiar.
lrPipeline = Pipeline()

lrPipeline.setStages([vectorizer, lr])

# Let's first train on the entire dataset to see what we get
lrModel = lrPipeline.fit(trainingSetDF)

# The intercept is as follows:
intercept = lrModel.stages[1].intercept

# The coefficents (i.e., weights) are as follows:
weights = lrModel.stages[1].coefficients

# Create a list of the column names (without PE)
featuresNoLabel = [col for col in datasetDF.columns if col != "PE"]

# Merge the weights and labels
coefficents = zip(weights, featuresNoLabel)

# Now let's sort the coefficients from greatest absolute weight most to the least absolute weight
coefficents.sort(key=lambda tup: abs(tup[0]), reverse=True)

equation = "y = {intercept}".format(intercept=intercept)
variables = []
for x in coefficents:
    weight = abs(x[0])
    name = x[1]
    symbol = "+" if (x[0] > 0) else "-"
    equation += (" {} ({} * {})".format(symbol, weight, name))

# Finally here is our equation
print("Linear Regression Equation: " + equation)

# Apply our LR model to the test data and predict power output
predictionsAndLabelsDF = lrModel.transform(testSetDF).select("AT", "V", "AP", "RH", "PE", "Predicted_PE")

display(predictionsAndLabelsDF)

# Now let's compute an evaluation metric for our test dataset
from pyspark.ml.evaluation import RegressionEvaluator

# Create an RMSE evaluator using the label and predicted columns
regEval = RegressionEvaluator(predictionCol="Predicted_PE", labelCol="PE", metricName="rmse")

# Run the evaluator on the DataFrame
rmse = regEval.evaluate(predictionsAndLabelsDF)

print("Root Mean Squared Error: %.2f" % rmse)

# Now let's compute another evaluation metric for our test dataset
r2 = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})

print("r2: {0:.2f}".format(r2))
# First we remove the table if it already exists
sqlContext.sql("DROP TABLE IF EXISTS Power_Plant_RMSE_Evaluation")
dbutils.fs.rm("dbfs:/user/hive/warehouse/Power_Plant_RMSE_Evaluation", True)

# Next we calculate the residual error and divide it by the RMSE
predictionsAndLabelsDF.selectExpr("PE", "Predicted_PE", "PE - Predicted_PE Residual_Error", "(PE - Predicted_PE) / {} Within_RSME".format(rmse)).registerTempTable("Power_Plant_RMSE_Evaluation")

%sql
SELECT * from Power_Plant_RMSE_Evaluation

%sql
-- Now we can display the RMSE as a Histogram
SELECT Within_RSME  from Power_Plant_RMSE_Evaluation

%sql
SELECT case when Within_RSME <= 1.0 AND Within_RSME >= -1.0 then 1
            when  Within_RSME <= 2.0 AND Within_RSME >= -2.0 then 2 else 3
       end RSME_Multiple, COUNT(*) AS count
FROM Power_Plant_RMSE_Evaluation
GROUP BY case when Within_RSME <= 1.0 AND Within_RSME >= -1.0 then 1  when  Within_RSME <= 2.0 AND Within_RSME >= -2.0 then 2 else 3 end

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# We can reuse the RegressionEvaluator, regEval, to judge the model based on the best Root Mean Squared Error
# Let's create our CrossValidator with 3 fold cross validation
crossval = CrossValidator(estimator=lrPipeline, evaluator=regEval, numFolds=3)

# Let's tune over our regularization parameter from 0.01 to 0.10
regParam = [x / 100.0 for x in range(1, 11)]

# We'll create a paramter grid using the ParamGridBuilder, and add the grid to the CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, regParam)
             .build())
crossval.setEstimatorParamMaps(paramGrid)

# Now let's find and return the best model
cvModel = crossval.fit(trainingSetDF).bestModel

# TODO: Replace <FILL_IN> with the appropriate code.
# Now let's use cvModel to compute an evaluation metric for our test dataset: testSetDF
predictionsAndLabelsDF = <FILL_IN>

# Run the previously created RMSE evaluator, regEval, on the predictionsAndLabelsDF DataFrame
rmseNew = <FILL_IN>

# Now let's compute the r2 evaluation metric for our test dataset
r2New = <FILL_IN>

print("Original Root Mean Squared Error: {0:2.2f}".format(rmse))
print("New Root Mean Squared Error: {0:2.2f}".format(rmseNew))
print("Old r2: {0:2.2f}".format(r2))
print("New r2: {0:2.2f}".format(r2New))

# TEST
Test.assertEquals(round(rmse, 2), 4.59, "Incorrect value for rmse")
Test.assertEquals(round(rmseNew, 2), 4.59, "Incorrect value for rmseNew")
Test.assertEquals(round(r2, 2), 0.93, "Incorrect value for r2")
Test.assertEquals(round(r2New, 2), 0.93, "Incorrect value for r2New")

print("Regularization parameter of the best model: {0:.2f}".format(cvModel.stages[-1]._java_obj.parent().getRegParam()))

# TODO: Replace <FILL_IN> with the appropriate code.
from pyspark.ml.regression import DecisionTreeRegressor

# Create a DecisionTreeRegressor
dt = <FILL_IN>

dt.setLabelCol("PE")\
  .setPredictionCol("Predicted_PE")\
  .setFeaturesCol("features")\
  .setMaxBins(100)

# Create a Pipeline
dtPipeline = <FILL_IN>

# Set the stages of the Pipeline
dtPipeline.<FILL_IN>

# TEST

Test.assertEqualsHashed(str(dtPipeline.getStages()[0].__class__.__name__), '4617be70bcf475326c0b07400b97b13457cc4949', "Incorrect pipeline stage 0")
Test.assertEqualsHashed(str(dtPipeline.getStages()[1].__class__.__name__), '46b18f257cf2f778d0d3b6e30ccc7b3398d7846a', "Incorrect pipeline stage 1")

# TODO: Replace <FILL_IN> with the appropriate code.
# Let's just reuse our CrossValidator with the new dtPipeline,  RegressionEvaluator regEval, and 3 fold cross validation
crossval.setEstimator(dtPipeline)

# Let's tune over our dt.maxDepth parameter on the values 2 and 3, create a paramter grid using the ParamGridBuilder
paramGrid = <FILL_IN>

# Add the grid to the CrossValidator
crossval.<FILL_IN>

# Now let's find and return the best model
dtModel = crossval.<FILL_IN>
