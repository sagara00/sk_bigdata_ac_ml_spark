
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
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
 

// csv파일에 header유무 확인, delimiter 확인
// Input file loading..
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "true").load("/Users/freeman/programming/dataset/iris/iris.data")

// schema 정보 및 data확인
df.printSchema

// 컬럼 개수 확인
df.schema.length

// 데이터 내용 흝어보기
df.show


// Training 과 Test 셋으로 Split Data (7:3비율)
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))


// Label 컬럼 및 Feature가 될만할 컬럼 파악
// 의미있는 컬럼만 추려서 Vector Assembler로 추린 컬럼들을 사용하여 feature컬럼 만들기
val columns = df.schema.map(a=>a.name).slice(0,4).toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

// LabelIndexer로 Label컬럼을 LabelIndex로 변경
val labelIndexer = new StringIndexer().setInputCol("_c4").setOutputCol("indexedLabel")

// FeatureIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)


// 알고리즘 취사선택
//val algorithm = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val algorithm = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)


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


