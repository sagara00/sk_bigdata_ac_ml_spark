
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.Normalizer
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor, LinearRegression, GBTRegressionModel, GBTRegressor, DecisionTreeRegressor}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types._
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.ml.{Pipeline, PipelineModel}

val sqlContext = new org.apache.spark.sql.SQLContext(sc)


// csv파일에 header유무 확인, delimiter 확인
// Input file loading
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("data/mllib/hour.csv")

// schema 정보 및 data확인
df.printSchema

// 컬럼 개수 확인
df.schema.length

// 데이터 내용 흝어보기
df.show


// Label 컬럼 및 Feature가 될만할 컬럼 파악 및 Label컬럼이 Int형임으로 Double로 바꾸어 주여야 한다.
val df2 = df.withColumn("cntTmp", df("cnt")cast(DoubleType)).drop("cnt").withColumnRenamed("cntTmp", "cnt")

// Training 과 Test 셋으로 Split Data (7:3비율)
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 11L)


// 의미있는 컬럼만 추려서 Vector Assembler로 추린 컬럼들을 사용하여 feature컬럼 만들기
val columns = df.schema.map(a=>a.name).slice(2,14).toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

// FeatureIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(24)


// 알고리즘 취사선택
//val algorithm = new LinearRegression().setLabelCol("cnt").setFeaturesCol("indexedFeatures")
//val algorithm = new DecisionTreeRegressor().setLabelCol("cnt").setFeaturesCol("indexedFeatures")
//val algorithm = new RandomForestRegressor().setLabelCol("cnt").setFeaturesCol("indexedFeatures")
val algorithm = new GBTRegressor().setLabelCol("cnt").setFeaturesCol("indexedFeatures").setMaxIter(30)


// pipeline 생성
val pipeline = new Pipeline().setStages(Array(assembler, featureIndexer, algorithm))

// import org.apache.spark.ml.feature.VectorIndexerModel
// val vimodel = model.stages(1).asInstanceOf[VectorIndexerModel]
// val categoricalFeatures: Set[Int] = vimodel.categoryMaps.keys.toSet
// println(s"Chose ${categoricalFeatures.size} categorical features: " + categoricalFeatures.mkString(", "))


// Training 
val model = pipeline.fit(trainingData)

// Test
val result = model.transform(testData)

// 결과보기
result.show


// Evaluator Define
val evaluator = new RegressionEvaluator().setLabelCol("cnt").setPredictionCol("prediction").setMetricName("rmse")
val eval_result = evaluator.evaluate(result)
println("Root Mean Squared Error (RMSE) on test data = " + eval_result)



