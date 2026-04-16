from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("PreviewHybrid").getOrCreate()

df = spark.read.parquet("hdfs:///user/up2552890/hybrid_results_parquet")

df.select(
    "parent_asin",
    "title",
    "semantic_score",
    "als_score",
    "final_score"
).orderBy(col("final_score").desc()) \
 .limit(20) \
 .coalesce(1) \
 .write.mode("overwrite") \
 .option("header", True) \
 .csv("hdfs:///user/up2552890/hybrid_preview_csv")

spark.stop()
