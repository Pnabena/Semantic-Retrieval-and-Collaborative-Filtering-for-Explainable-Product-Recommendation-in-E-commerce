from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("ALSBuildModel").getOrCreate()

als_input = spark.read.parquet("als_input_parquet")
item_lookup = spark.read.parquet("item_lookup_parquet")
user_lookup = spark.read.parquet("user_lookup_parquet")

train_df, test_df = als_input.randomSplit([0.8, 0.2], seed=42)

als = ALS(
    userCol="user_idx",
    itemCol="item_idx",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    implicitPrefs=False,
    rank=10,
    maxIter=10,
    regParam=0.1
)

als_model = als.fit(train_df)

predictions = als_model.transform(test_df)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print("RMSE:", rmse)

# Save the trained model
als_model.write().overwrite().save("hdfs:///user/up2552890/als_model")

# Optional sanity-check recommendations
sample_users = als_input.select("user_idx").distinct().limit(500)

user_recs_subset = als_model.recommendForUserSubset(sample_users, 10)

user_recs_flat = user_recs_subset.select(
    col("user_idx"),
    explode(col("recommendations")).alias("rec")
).select(
    col("user_idx"),
    col("rec.item_idx").alias("item_idx"),
    col("rec.rating").alias("als_score")
)

als_results = user_recs_flat.join(
    item_lookup, on="item_idx", how="left"
).join(
    user_lookup, on="user_idx", how="left"
)

als_results.write.mode("overwrite").parquet("hdfs:///user/up2552890/als_results_parquet")

spark.stop()
