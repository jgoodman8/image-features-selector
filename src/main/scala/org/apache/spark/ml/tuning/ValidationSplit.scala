package org.apache.spark.ml.tuning

import java.util.{Locale, List => JList}

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.shared.{HasCollectSubModels, HasParallelism}
import org.apache.spark.ml.param.{DatasetParam, ParamMap, Params}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.util.ThreadUtils
import org.json4s.DefaultFormats

import scala.collection.JavaConverters._
import scala.concurrent.Future
import scala.concurrent.duration.Duration
import scala.language.existentials

/**
  * Params for [[TrainValidationSplit]] and [[TrainValidationSplitModel]].
  */
private[ml] trait ValidationSplitParams extends ValidatorParams {

  /** @group param */
  val validationSet: DatasetParam = new DatasetParam(this, "validationSet", "")

  /** @group getParam */
  def getValidationSet: Dataset[_] = $(validationSet)

  setDefault(validationSet -> null)
}

/**
  * Validation for hyper-parameter tuning.
  * Uses evaluation metric on the validation set to select the best model.
  * Similar to [[TrainValidationSplit]], but sets are specified.
  */
@Since("1.5.0")
class ValidationSplit @Since("1.5.0")(@Since("1.5.0") override val uid: String)
  extends Estimator[ValidationSplitModel]
    with ValidationSplitParams with HasParallelism with HasCollectSubModels // with ValidatorParams
    with MLWritable with Logging {

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("tvs"))

  /** @group setParam */
  @Since("1.2.0")
  def setEstimator(value: Estimator[_]): this.type = set(estimator, value)

  /** @group setParam */
  @Since("1.2.0")
  def setEstimatorParamMaps(value: Array[ParamMap]): this.type = set(estimatorParamMaps, value)

  /** @group setParam */
  @Since("1.2.0")
  def setEvaluator(value: Evaluator): this.type = set(evaluator, value)

  /** @group setParam */
  @Since("1.2.0")
  def setValidationSet(value: Dataset[_]): this.type = set(validationSet, value)

  /** @group setParam */
  @Since("2.0.0")
  def setSeed(value: Long): this.type = set(seed, value)

  /**
    * Set the maximum level of parallelism to evaluate models in parallel.
    * Default is 1 for serial evaluation
    *
    * @group expertSetParam
    */
  @Since("2.3.0")
  def setParallelism(value: Int): this.type = set(parallelism, value)

  /**
    * Whether to collect submodels when fitting. If set, we can get submodels from
    * the returned model.
    *
    * Note: If set this param, when you save the returned model, you can set an option
    * "persistSubModels" to be "true" before saving, in order to save these submodels.
    * You can check documents of
    * {@link org.apache.spark.ml.tuning.ValidationSplitModel.ValidationSplitModelWriter}
    * for more information.
    *
    * @group expertSetParam
    */
  @Since("2.3.0")
  def setCollectSubModels(value: Boolean): this.type = set(collectSubModels, value)

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): ValidationSplitModel = instrumented { instr =>
    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)
    val validationDataset = $(validationSet)

    // Create execution context based on $(parallelism)
    val executionContext = getExecutionContext

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logDataset(validationDataset)
    instr.logParams(this, seed, parallelism)
    logTuningParams(instr)

    dataset.cache()
    validationDataset.cache()

    val collectSubModelsParam = $(collectSubModels)

    var subModels: Option[Array[Model[_]]] = if (collectSubModelsParam) {
      Some(Array.fill[Model[_]](epm.length)(null))
    } else None

    // Fit models in a Future for training in parallel
    instr.logDebug(s"Train split with multiple sets of parameters.")
    val metricFutures = epm.zipWithIndex.map { case (paramMap, paramIndex) =>
      Future[Double] {
        val model = est.fit(dataset, paramMap).asInstanceOf[Model[_]]

        if (collectSubModelsParam) {
          subModels.get(paramIndex) = model
        }
        // TODO: duplicate evaluator to take extra params from input
        val metric = eval.evaluate(model.transform(validationDataset, paramMap))
        instr.logDebug(s"Got metric $metric for model trained with $paramMap.")
        metric
      }(executionContext)
    }

    // Wait for all metrics to be calculated
    val metrics = metricFutures.map(ThreadUtils.awaitResult(_, Duration.Inf))

    // Unpersist training & validation set once all metrics have been produced
    dataset.unpersist()
    validationDataset.unpersist()

    instr.logInfo(s"Train validation split metrics: ${metrics.toSeq}")
    val (bestMetric, bestIndex) =
      if (eval.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)
    instr.logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    instr.logInfo(s"Best train validation split metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
    copyValues(new ValidationSplitModel(uid, bestModel, metrics)
      .setSubModels(subModels).setParent(this))
  }

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = transformSchemaImpl(schema)

  @Since("1.4.0")
  override def copy(extra: ParamMap): ValidationSplit = {
    val copied = defaultCopy(extra).asInstanceOf[ValidationSplit]
    if (copied.isDefined(estimator)) {
      copied.setEstimator(copied.getEstimator.copy(extra))
    }
    if (copied.isDefined(evaluator)) {
      copied.setEvaluator(copied.getEvaluator.copy(extra))
    }
    copied
  }

  @Since("1.6.0")
  override def write: MLWriter = new ValidationSplit.ValidatorWriter(this)
}

@Since("1.6.0")
object ValidationSplit extends MLReadable[ValidationSplit] {

  @Since("1.6.02")
  override def read: MLReader[ValidationSplit] = new ValidationSplitReader

  @Since("1.6.0")
  override def load(path: String): ValidationSplit = super.load(path)

  private[ValidationSplit] class ValidatorWriter(instance: ValidationSplit) extends MLWriter {

    ValidatorParams.validateParams(instance)

    override protected def saveImpl(path: String): Unit =
      ValidatorParams.saveImpl(path, instance, sc)
  }

  private class ValidationSplitReader extends MLReader[ValidationSplit] {

    /** Checked against metadata when loading model */
    private val className = classOf[ValidationSplit].getName

    override def load(path: String): ValidationSplit = {
      implicit val format = DefaultFormats

      val (metadata, estimator, evaluator, estimatorParamMaps) =
        ValidatorParams.loadImpl(path, sc, className)
      val vs = new ValidationSplit(metadata.uid)
        .setEstimator(estimator)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(estimatorParamMaps)
      metadata.getAndSetParams(vs, skipParams = Option(List("estimatorParamMaps")))
      vs
    }
  }

}

/**
  * Model from train validation split.
  *
  * @param uid               Id.
  * @param bestModel         Estimator determined best model.
  * @param validationMetrics Evaluated validation metrics.
  */
@Since("1.2.0")
class ValidationSplitModel private[ml](
                                        @Since("1.5.0") override val uid: String,
                                        @Since("1.5.0") val bestModel: Model[_],
                                        @Since("1.5.0") val validationMetrics: Array[Double])
  extends Model[ValidationSplitModel] with ValidationSplitParams with MLWritable {

  /** A Python-friendly auxiliary constructor. */
  private[ml] def this(uid: String, bestModel: Model[_], validationMetrics: JList[Double]) = {
    this(uid, bestModel, validationMetrics.asScala.toArray)
  }

  private var _subModels: Option[Array[Model[_]]] = None

  private[tuning] def setSubModels(subModels: Option[Array[Model[_]]])
  : ValidationSplitModel = {
    _subModels = subModels
    this
  }

  // A Python-friendly auxiliary method
  private[tuning] def setSubModels(subModels: JList[Model[_]])
  : ValidationSplitModel = {
    _subModels = if (subModels != null) {
      Some(subModels.asScala.toArray)
    } else {
      None
    }
    this
  }

  /**
    * @return submodels represented in array. The index of array corresponds to the ordering of
    *         estimatorParamMaps
    * @throws IllegalArgumentException if subModels are not available. To retrieve subModels,
    *                                  make sure to set collectSubModels to true before fitting.
    */
  @Since("2.3.0")
  def subModels: Array[Model[_]] = {
    require(_subModels.isDefined, "subModels not available, To retrieve subModels, make sure " +
      "to set collectSubModels to true before fitting.")
    _subModels.get
  }

  @Since("2.3.0")
  def hasSubModels: Boolean = _subModels.isDefined

  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    bestModel.transform(dataset)
  }

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = {
    bestModel.transformSchema(schema)
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): ValidationSplitModel = {
    val copied = new ValidationSplitModel(
      uid,
      bestModel.copy(extra).asInstanceOf[Model[_]],
      validationMetrics.clone()
    ).setSubModels(ValidationSplitModel.copySubModels(_subModels))
    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: ValidationSplitModel.ValidationSplitModelWriter = {
    new ValidationSplitModel.ValidationSplitModelWriter(this)
  }
}

@Since("1.6.0")
object ValidationSplitModel extends MLReadable[ValidationSplitModel] {

  private[ValidationSplitModel] def copySubModels(subModels: Option[Array[Model[_]]])
  : Option[Array[Model[_]]] = {
    subModels.map(_.map(_.copy(ParamMap.empty).asInstanceOf[Model[_]]))
  }

  @Since("1.6.0")
  override def read: MLReader[ValidationSplitModel] = new ValidationSplitModelReader

  @Since("1.6.0")
  override def load(path: String): ValidationSplitModel = super.load(path)

  /**
    * Writer for TrainValidationSplitModel.
    *
    * @param instance TrainValidationSplitModel instance used to construct the writer
    *
    *                 TrainValidationSplitModel supports an option "persistSubModels", with possible values
    *                 "true" or "false". If you set the collectSubModels Param before fitting, then you can
    *                 set "persistSubModels" to "true" in order to persist the subModels. By default,
    *                 "persistSubModels" will be "true" when subModels are available and "false" otherwise.
    *                 If subModels are not available, then setting "persistSubModels" to "true" will cause
    *                 an exception.
    */
  @Since("2.3.0")
  final class ValidationSplitModelWriter private[tuning](instance: ValidationSplitModel) extends MLWriter {

    ValidatorParams.validateParams(instance)

    override protected def saveImpl(path: String): Unit = {
      val persistSubModelsParam = optionMap.getOrElse("persistsubmodels",
        if (instance.hasSubModels) "true" else "false")

      require(Array("true", "false").contains(persistSubModelsParam.toLowerCase(Locale.ROOT)),
        s"persistSubModels option value ${persistSubModelsParam} is invalid, the possible " +
          "values are \"true\" or \"false\"")
      val persistSubModels = persistSubModelsParam.toBoolean

      import org.json4s.JsonDSL._
      val extraMetadata = ("validationMetrics" -> instance.validationMetrics.toSeq) ~
        ("persistSubModels" -> persistSubModels)
      ValidatorParams.saveImpl(path, instance, sc, Some(extraMetadata))
      val bestModelPath = new Path(path, "bestModel").toString
      instance.bestModel.asInstanceOf[MLWritable].save(bestModelPath)
      if (persistSubModels) {
        require(instance.hasSubModels, "When persisting tuning models, you can only set " +
          "persistSubModels to true if the tuning was done with collectSubModels set to true. " +
          "To save the sub-models, try rerunning fitting with collectSubModels set to true.")
        val subModelsPath = new Path(path, "subModels")
        for (paramIndex <- 0 until instance.getEstimatorParamMaps.length) {
          val modelPath = new Path(subModelsPath, paramIndex.toString).toString
          instance.subModels(paramIndex).asInstanceOf[MLWritable].save(modelPath)
        }
      }
    }
  }

  private class ValidationSplitModelReader extends MLReader[ValidationSplitModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[ValidationSplitModel].getName

    override def load(path: String): ValidationSplitModel = {
      implicit val format = DefaultFormats

      val (metadata, estimator, evaluator, estimatorParamMaps) =
        ValidatorParams.loadImpl(path, sc, className)
      val bestModelPath = new Path(path, "bestModel").toString
      val bestModel = DefaultParamsReader.loadParamsInstance[Model[_]](bestModelPath, sc)
      val validationMetrics = (metadata.metadata \ "validationMetrics").extract[Seq[Double]].toArray
      val persistSubModels = (metadata.metadata \ "persistSubModels")
        .extractOrElse[Boolean](false)

      val subModels: Option[Array[Model[_]]] = if (persistSubModels) {
        val subModelsPath = new Path(path, "subModels")
        val _subModels = Array.fill[Model[_]](estimatorParamMaps.length)(null)
        for (paramIndex <- 0 until estimatorParamMaps.length) {
          val modelPath = new Path(subModelsPath, paramIndex.toString).toString
          _subModels(paramIndex) =
            DefaultParamsReader.loadParamsInstance(modelPath, sc)
        }
        Some(_subModels)
      } else None

      val model = new ValidationSplitModel(metadata.uid, bestModel, validationMetrics)
        .setSubModels(subModels)
      model.set(model.estimator, estimator)
        .set(model.evaluator, evaluator)
        .set(model.estimatorParamMaps, estimatorParamMaps)
      metadata.getAndSetParams(model, skipParams = Option(List("estimatorParamMaps")))
      model
    }
  }

}
