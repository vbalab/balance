import io
import pandas as pd

from os import system
from contextlib import closing
from webdataset.gopen import gopen_pipe
from getpass import getuser
from re import sub
from typing import Literal
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.column import Column
from pyspark.sql import functions as f


def dec_detect(col: str, col_type: str) -> Column:
    dec_flag = col_type[:8] == "decimal("
    decint_flag = col_type[-3:] == ",0)"
    if dec_flag:
        if decint_flag:
            return f.col(col).cast("integer")
        else:
            return f.col(col).cast("double")
    else:
        return f.col(col)


def convert_decimals(spark_df: DataFrame) -> DataFrame:
    select_expr = [dec_detect(c, t) for c, t in spark_df.dtypes]
    return spark_df.select(select_expr)


def hadoop_open(path):
    hadoop_pipe = f"pipe:hadoop fs -cat {path}/*"
    with closing(gopen_pipe(hadoop_pipe)) as input_stream:
        file_bytes = input_stream.read()
    return io.BytesIO(file_bytes)


def spark_df_toPandas(
    spark_df: DataFrame,
    read_cache: bool = False,
    write_cache: bool = False,
    convert_decimals: bool = True,
    infer_method: Literal[None, "infer_objects", "convert_dtypes"] = None,
) -> pd.DataFrame:
    """This function reproduces .toPandas method of Spark DataFrame

    It saves a DataFrame as a parquet to hdfs and then reads it back to Pandas

    Parameters
    ----------
    spark_df : pyspark.sql.dataframe.DataFrame
        A Spark DataFrame to convert into pd.DataFrame
    read_cache : bool, optional
        If True, uses the cached file if it is avaliable for this spark_df,
        by default False
    write_cache : bool, optional
        If False, deletes the cached file after reading it,
        by default False
    convert_decimals : bool, optional
        If True, converts all decimal types to doubles or ints
        (depending on decimal precision)
        by default True
    infer_method : str, optional
        A method to infer datatypes in a pd.DataFrame.
        Avaliable options are [None, 'infer_objects', 'convert_dtypes'],
        by default None

    Returns
    -------
    pd.DataFrame
        The resulting pd.DataFrame
    """

    # computing hash of a spark_df's logical plan in order to have
    # the same file name for the same spark_df
    plan_hash = hash(
        sub(
            "\#[0-9]*",
            "",
            str(spark_df._jdf.queryExecution().logical()),
        )
    )

    hadoop_path = f"hdfs://adh-users/user/{getuser()}/data/{plan_hash}.parquet"

    # this converts decimals to doubles or integers if convert_decimals==True
    if convert_decimals:
        select_expr = [dec_detect(c, t) for c, t in spark_df.dtypes]
    else:
        select_expr = "*"  # [f.col(c) for c, t in spark_df.dtypes]

    # this if loop ensures that the data comes from cache if and only if read_cache==True and there is cache for this spark_df
    if read_cache:
        code = system(f"hdfs dfs -test -e {hadoop_path}")
        if code != 0:
            spark_df.select(select_expr).repartition(1).write.mode("overwrite").parquet(
                hadoop_path
            )
        else:
            pass
    else:
        spark_df.select(select_expr).repartition(1).write.mode("overwrite").parquet(
            hadoop_path
        )

    # this part reads the file
    with closing(hadoop_open(hadoop_path)) as bytes_stream:
        table = pd.read_parquet(bytes_stream)

    # remove file if write_cache == False
    if write_cache:
        pass
    else:
        system(f"hdfs dfs -rm -r {hadoop_path}")

    # apply postprocessing on pd.DataFrame
    if infer_method:
        table = table.__getattr__(infer_method)()

    return table
