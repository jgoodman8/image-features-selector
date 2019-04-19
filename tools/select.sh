#!/bin/bash

NUM_EXECUTORS=20
EXECUTOR_CORES=8
EXECUTOR_MEMORY=36g
DRIVER_CORES=8
DRIVER_MEMORY=36g

CLASS=ifs.jobs.FeatureSelectionPipeline
JAR=image-features-selector/target/scala-2.11/images-features-selection-assembly-0.1.jar
TRAIN_FILE=datasets/train_tiny_features_ensemble_1536_v6.csv
TEST_FILE=datasets/test_tiny_features_ensemble_1536_v6.csv
OUTPUT_FILE=datasets/tiny_selected_features
QUEUE=default
METHOD=${1:-chisq}
NUM_FEATURES=${2:-20}
APP_NAME=IFS_${METHOD}_${NUM_FEATURES}

spark-submit --name $APP_NAME --queue $QUEUE --master yarn --deploy-mode cluster --num-executors $NUM_EXECUTORS --executor-cores $EXECUTOR_CORES --executor-memory $EXECUTOR_MEMORY --driver-cores $DRIVER_CORES --driver-memory $DRIVER_MEMORY --conf "spark.driver.maxResultSize=10g" --conf "spark.driver.maxResultSize=3g" --conf "spark.kryoserializer.buffer.max=640m" --class $CLASS $JAR $APP_NAME $TRAIN_FILE $TEST_FILE $OUTPUT_FILE $METHOD $NUM_FEATURES