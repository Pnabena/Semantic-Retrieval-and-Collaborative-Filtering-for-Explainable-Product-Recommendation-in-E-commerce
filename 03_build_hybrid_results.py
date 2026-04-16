from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, min as spark_min, max as spark_max
from pyspark.ml.recommendation import ALSModel

spark = SparkSession.builder.appName("BuildHybridResults").getOrCreate()

# Load inputs
semantic_df = spark.read.parquet("hdfs:///user/up2552890/semantic_results_parquet")
item_lookup = spark.read.parquet("hdfs:///user/up2552890/item_lookup_parquet")
user_lookup = spark.read.parquet("hdfs:///user/up2552890/user_lookup_parquet")
als_model = ALSModel.load("hdfs:///user/up2552890/als_model")

# Pick demo user
test_user = "AE22XCKLYXP5YZF6QSFXUD3KM5KA"

# Get the user's ALS index
user_row = user_lookup.filter(col("user_id") == test_user).select("user_id", "user_idx")

# Keep only semantic products that exist in ALS item space
semantic_items = semantic_df.join(
    item_lookup.select("parent_asin", "item_idx"),
    on="parent_asin",
    how="inner"
)

# Attach the user to every semantic candidate
candidate_df = semantic_items.crossJoin(user_row)

# Score semantic candidates with ALS
als_scored = als_model.transform(candidate_df)

# Rename prediction to als_score
hybrid_df = als_scored.withColumnRenamed("prediction", "als_score")

# Fill null ALS scores if needed
hybrid_df = hybrid_df.fillna({"als_score": 0.0})

# Normalize semantic score
sem_stats = hybrid_df.agg(
    spark_min("semantic_score").alias("sem_min"),
    spark_max("semantic_score").alias("sem_max")
).collect()[0]

# Normalize ALS score
als_stats = hybrid_df.agg(
    spark_min("als_score").alias("als_min"),
    spark_max("als_score").alias("als_max")
).collect()[0]

sem_min, sem_max = sem_stats["sem_min"], sem_stats["sem_max"]
als_min, als_max = als_stats["als_min"], als_stats["als_max"]

hybrid_df = hybrid_df.withColumn(
    "semantic_norm",
    (col("semantic_score") - lit(sem_min)) / (lit((sem_max - sem_min) if sem_max != sem_min else 1.0))
).withColumn(
    "als_norm",
    (col("als_score") - lit(als_min)) / (lit((als_max - als_min) if als_max != als_min else 1.0))
)

# Weighted hybrid score
alpha = 0.7
beta = 0.3

hybrid_df = hybrid_df.withColumn(
    "final_score",
    alpha * col("semantic_norm") + beta * col("als_norm")
)

# Save ranked results
final_df = hybrid_df.orderBy(col("final_score").desc())

final_df.write.mode("overwrite").parquet(
    "hdfs:///user/up2552890/hybrid_results_parquet"
)

spark.stop()