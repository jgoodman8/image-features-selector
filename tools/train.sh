#!/bin/bash

NUM_EXECUTORS=40
EXECUTOR_CORES=6
EXECUTOR_MEMORY=36g
DRIVER_CORES=8
DRIVER_MEMORY=36g

CLASS=ifs.jobs.ClassificationPipeline
JAR=image-features-selector/target/scala-2.11/images-features-selection-assembly-0.1.jar
QUEUE=default
METHOD=$1
TRAIN_FILE=datasets/tiny_selected_features/$2
TEST_FILE=datasets/tiny_selected_features/$3
APP_NAME=IClass_${METHOD}_${2}_${3}

spark-submit --name $APP_NAME --queue $QUEUE --master yarn --deploy-mode cluster --num-executors $NUM_EXECUTORS --executor-cores $EXECUTOR_CORES --executor-memory $EXECUTOR_MEMORY --driver-cores $DRIVER_CORES --driver-memory $DRIVER_MEMORY --conf "spark.driver.maxResultSize=10g" --conf "spark.driver.maxResultSize=3g" --conf "spark.kryoserializer.buffer.max=640m" --class $CLASS $JAR $APP_NAME $TRAIN_FILE $TEST_FILE $METHOD
