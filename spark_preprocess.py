import pandas as pd
from pathlib import PurePosixPath
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType

INPUT_PATH_1 = "s3a://raw-wildlife/source_1/*/*"
INPUT_PATH_2 = "s3a://raw-wildlife/source_2/*/*"
OUTPUT_PATH = "s3a://processed-wildlife/"

CLASS_MAPPING = {
    'Urs': 'Bear', 'Cerb Comun': 'Deer', 'Mistret': 'Boar',
    'SunBear': 'Bear', 'Elephant': 'Elephant', 'Tiger': 'Tiger',
    'Tapir': 'Tapir', 'ForestBG': 'Background'
}

spark = SparkSession.builder \
    .appName("WildlifeSimple") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


def get_label_udf(image_path: pd.Series) -> pd.Series:
    labels = []
    for path in image_path:
        try:
            parts = [p for p in str(path).split("/") if p]
            class_name = parts[-2] if len(parts) >= 2 else "Unknown"
            label = CLASS_MAPPING.get(class_name, class_name)
            labels.append(label if label not in ["", None] else "Unknown")
        except:
            labels.append("Unknown")
    return pd.Series(labels)


get_label_pandas_udf = pandas_udf(get_label_udf, returnType=StringType())

print("Wildlife Pipeline")

df1 = spark.read.format("image").load(INPUT_PATH_1)
df2 = spark.read.format("image").load(INPUT_PATH_2)
df_raw = df1.union(df2)
print(f"Прочитано: {df_raw.count()} изображений")

df_with_label = df_raw.withColumn("label", get_label_pandas_udf(col("image.origin")))

print("\nКлассы:")
df_with_label.groupBy("label").count().orderBy("count", ascending=False).show()

print(f"\nСохранение в {OUTPUT_PATH}...")
df_out = df_with_label.select("label", col("image.data").alias("image_bytes"))

df_out.write \
    .mode("overwrite") \
    .partitionBy("label") \
    .parquet(OUTPUT_PATH)

spark.stop()
