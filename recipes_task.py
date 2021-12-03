import datetime
import os
import logging
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType


class ISODurationMixin:

    @staticmethod
    def get_iso_split(s, split):
        if split in s:
            n, s = s.split(split)
        else:
            n = 0
        return n, s

    @staticmethod
    def parse_duration_to_seconds(duration) -> int:

        s = duration.split('P')[-1]

        days, s = ISODurationMixin.get_iso_split(s, 'D')
        _, s = ISODurationMixin.get_iso_split(s, 'T')
        hours, s = ISODurationMixin.get_iso_split(s, 'H')
        minutes, s = ISODurationMixin.get_iso_split(s, 'M')
        seconds, s = ISODurationMixin.get_iso_split(s, 'S')

        dt = datetime.timedelta(
            days=int(days),
            hours=int(hours),
            minutes=int(minutes),
            seconds=int(seconds)
        )
        return int(dt.total_seconds())

    @staticmethod
    def parse_duration_to_minutes(duration) -> int:
        seconds = ISODurationMixin.parse_duration_to_seconds(duration)
        return int(seconds / 60)


class RecipesJob:

    def __init__(self, source_path, target_path):
        self.source_path = source_path
        self.target_path = target_path
        self.spark = self._init_spark()
    
    def __call__(self):
        self.execute()

    @staticmethod
    def _init_spark() -> "SparkSession":
        master = os.getenv("SPARK_MASTER", "local[*]")
        spark = (
            SparkSession.builder
            .appName("RecipesJob")
            .master(master)
            .config("spark.default.parallelism", 100)
            .getOrCreate()
        )

        return spark

    def load_data(self) -> "DataFrame":
        logging.info(f"Fetching data from: {self.source_path}")
        source_df = self.spark.read.format("json").load(self.source_path)

        return source_df

    def persist_data(self, df: "DataFrame"):
        logging.info(f"Persisting data to: {self.target_path}")
        df.write.format("csv").mode("overwrite").option("header", True).save(self.target_path)

    def filter_by_ingredients(self, df: "DataFrame") -> "DataFrame":
        assert all(col in df.columns for col in ["ingredients"])

        filtered_df = df.filter(F.lower(F.col("ingredients")).like("%beef%"))

        return filtered_df

    def clean_durations(self, df: "DataFrame") -> "DataFrame":
        assert all(col in df.columns for col in ["cookTime", "prepTime"])

        duration_to_minutes_udf = F.udf(
            lambda duration: ISODurationMixin.parse_duration_to_minutes(duration),
            IntegerType()
        )
        cleaned_df = (
            df
            .withColumn("cook_time_minutes", duration_to_minutes_udf(F.col("cookTime")))
            .withColumn("prep_time_minutes", duration_to_minutes_udf(F.col("prepTime")))
        )

        return cleaned_df

    def set_difficulty(self, df: "DataFrame") -> "DataFrame":
        assert all(col in df.columns for col in ["cook_time_minutes", "prep_time_minutes"])

        prepared_df = (
            df
            .withColumn("total_cook_time", F.col("cook_time_minutes") + F.col("prep_time_minutes"))
            .withColumn(
                "difficulty",
                F.when(F.col("total_cook_time") <= 30, "easy")
                .when((F.col("total_cook_time") > 30) & (F.col("total_cook_time") <= 60), "medium")
                .when(F.col("total_cook_time") > 60, "hard")
                .otherwise("Unknown")
            )
        )

        return prepared_df

    def aggregate_per_difficulty(self, df: "DataFrame") -> "DataFrame":
        assert all(col in df.columns for col in ["difficulty", "total_cook_time"])

        aggregated_df = (
            df
            .groupBy("difficulty")
            .agg(
                F.avg("total_cook_time").cast(DoubleType()).alias("avg_total_cooking_time")
            )
        )

        return aggregated_df

    def execute(self):
        raw_df = self.load_data()
        filtered_df = self.filter_by_ingredients(raw_df)
        cleaned_df = self.clean_durations(filtered_df)
        df_with_difficulty = self.set_difficulty(cleaned_df)
        aggregated_df = self.aggregate_per_difficulty(df_with_difficulty)
        self.persist_data(aggregated_df)


# entry point for airflow task
def run_recipes_job(source_path, target_path):
    job = RecipesJob(source_path, target_path)
    job()
