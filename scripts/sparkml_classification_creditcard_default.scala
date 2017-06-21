
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.Normalizer
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, BinaryLogisticRegressionSummary}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}


val sqlContext = new org.apache.spark.sql.SQLContext(sc)


// csv파일에 header유무 확인, delimiter 확인
// Input file loading
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("data/mllib/creditCardDefault.csv")

// schema 정보 및 data확인
df.printSchema

// 컬럼 개수 확인
df.schema.length

// 데이터 내용 흝어보기
df.show


// Training 과 Test 셋으로 Split Data (7:3비율)
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 11L)


// Label 컬럼 및 Feature가 될만할 컬럼 파악
// 의미있는 컬럼만 추려서 Vector Assembler로 추린 컬럼들을 사용하여 feature컬럼 만들기
val columns = df.schema.map(a=>a.name).slice(1,24).toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

// LabelIndexer로 Label컬럼을 LabelIndex로 변경
val labelIndexer = new StringIndexer().setInputCol("default_payment_next_month").setOutputCol("indexedLabel")

//val encoder = new OneHotEncoder().setInputCol("indexedLabel2").setOutputCol("indexedLabel")

// FeatureIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(8)


// 알고리즘 취사선택
val algorithm = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//val algorithm = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//val algorithm = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
//val algorithm = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10)


// Label Converter
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.fit(trainingData).labels)

// Pipeline 만들기
val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, featureIndexer, algorithm, labelConverter))

// Training 
val model = pipeline.fit(trainingData)

// Test
val result = model.transform(testData)

// 결과 보기
result.printSchema
result.show


// Evaluation
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(result)
println("Test Error = " + (1.0 - accuracy))


// Logistic Regression Model Summary
// pipeline의 4번째 구성요소 접근
val alg_model = model.stages(3).asInstanceOf[LogisticRegressionModel]
val lr_summary = alg_model.summary
val objectiveHistory = lr_summary.objectiveHistory
objectiveHistory.foreach(loss => println(loss))


val binarySummary = lr_summary.asInstanceOf[BinaryLogisticRegressionSummary]
val roc = binarySummary.roc
roc.show()
println(binarySummary.areaUnderROC)




// ****************************** Cross-Validation ************************* //

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("data/mllib/creditCardDefault.csv")
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 11L)

val columns = df.schema.map(a=>a.name).slice(1,24).toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")
val labelIndexer = new StringIndexer().setInputCol("default_payment_next_month").setOutputCol("indexedLabel")
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)
val algorithm = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.fit(trainingData).labels)
val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, featureIndexer, algorithm, labelConverter))

// For cross validation
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val paramGrid = new ParamGridBuilder().addGrid(algorithm.regParam, Array(0.1, 0.01)).addGrid(algorithm.fitIntercept).addGrid(algorithm.elasticNetParam, Array(0.0, 0.5, 1.0)).build()
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)

// Training
val model = cv.fit(trainingData)

// Test
val result = model.transform(testData)

// Evaluation
val accuracy = evaluator.evaluate(result)
println("Test Error = " + (1.0 - accuracy))



