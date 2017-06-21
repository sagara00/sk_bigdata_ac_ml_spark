
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.Normalizer
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, BinaryLogisticRegressionSummary}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}


val sqlContext = new org.apache.spark.sql.SQLContext(sc)

// Input file loading
//val train = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("data/mllib/titanic_train.csv")
//val test = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("data/mllib/titanic_test.csv")

val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("data/mllib/titanic_train.csv")
val Array(train, test) = df.randomSplit(Array(0.7, 0.3), seed = 11L)


// Print Data Schema
train.printSchema
// |-- PassengerId: integer (nullable = true)
// |-- Survived: integer (nullable = true)
// |-- Pclass: integer (nullable = true)
// |-- Name: string (nullable = true)
// |-- Sex: string (nullable = true)
// |-- Age: double (nullable = true)
// |-- SibSp: integer (nullable = true)
// |-- Parch: integer (nullable = true)
// |-- Ticket: string (nullable = true)
// |-- Fare: double (nullable = true)
// |-- Cabin: string (nullable = true)
// |-- Embarked: string (nullable = true)


// na value remove if needed
//val train = train.na.drop()


// 전체 컬럼에 대해서 na 값 찾기
train.schema.map(a=>a.name).foreach{name=>println("Name: " + name); train.filter(train(name).isNull || train(name) === "").show}
// => Age, Cabin, Embarked 에 na 존재


// column count show
println("Total row count: " + train.count)
train.schema.map(a=>a.name).foreach{name=> train.groupBy(train(name)).count.sort(desc("count")).show}
// => 전체 row수는 891

// ********** Feature Selection ********** //
// Age : null count - 177 --> 평균값으로 대체 
// Cabin: "" count - 687  --> 전체 개수에 비해 너무 많아서 그냥 무시
// Embarked: "" count - 2 --> 가장 많은 빈도로 나타난 값으로 대체


// replace missing value with the average in Age column
val avg_age = train.select(mean("Age")).first()(0).asInstanceOf[Double]
val train2 = train.na.fill(avg_age, Seq("Age"))


// what about "" (empty string).. just use udf
val replaceEmpty = sqlContext.udf.register("replaceEmpty", (embarked: String) => {if (embarked  == null) "S" else embarked })
val train3 = train2.withColumn("Embarked", replaceEmpty(train2("Embarked")))

// check the result
train3.groupBy(train3("Embarked")).count.sort(desc("count")).show



// adding some useful features.. using user-defined function
val addChild = sqlContext.udf.register("addChild", (sex: String, age: Double) => {if (age < 15) "Child" else sex })
val withFamily = sqlContext.udf.register("withFamily", (sib: Int, par: Int) => {if (sib + par > 3) 1.0 else 0.0 })

val train4 = train3.withColumn("Sex", addChild(train3("Sex"), train3("Age")))
val train5 = train4.withColumn("Family", withFamily(train4("SibSp"), train4("Parch")))


// Check the schema
train5.printSchema
// |-- PassengerId: integer (nullable = true)
// |-- Survived: integer (nullable = true)
// |-- Pclass: integer (nullable = true)
// |-- Name: string (nullable = true)
// |-- Sex: string (nullable = true)
// |-- Age: double (nullable = false)
// |-- SibSp: integer (nullable = true)
// |-- Parch: integer (nullable = true)
// |-- Ticket: string (nullable = true)
// |-- Fare: double (nullable = true)
// |-- Cabin: string (nullable = true)
// |-- Embarked: string (nullable = true)
// |-- Family: double (nullable = true)


// Use StringIndexer on String schema type
val sexIndex = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
val embarkedIndex = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")
val survivedIndex = new StringIndexer().setInputCol("Survived").setOutputCol("SurvivedIndex")

//val encoder1 = new OneHotEncoder().setInputCol("SexIndex2").setOutputCol("SexIndex")
//val encoder2 = new OneHotEncoder().setInputCol("EmbarkedIndex2").setOutputCol("EmbarkedIndex")


// Select columns and Vector Assembler
val columns = Seq("Pclass", "SexIndex", "Age", "Fare", "EmbarkedIndex", "Family").toArray
val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")

// Normalizer
//val normalizer = new Normalizer().setInputCol("features_temp").setOutputCol("features")

// FeatureIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(8)

// Classifiers
//val classmodel = new RandomForestClassifier().setLabelCol("SurvivedIndex").setFeaturesCol("indexedFeatures").setNumTrees(10)
//val classmodel = new GBTClassifier().setLabelCol("SurvivedIndex").setFeaturesCol("indexedFeatures").setMaxIter(10)
//val classmodel = new DecisionTreeClassifier().setLabelCol("SurvivedIndex").setFeaturesCol("indexedFeatures")
val classmodel = new LogisticRegression().setLabelCol("SurvivedIndex").setFeaturesCol("indexedFeatures")

// LabelConverter
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(survivedIndex.fit(train).labels)

// Pipeline 
//val pipeline = new Pipeline().setStages(Array(sexIndex, embarkedIndex, survivedIndex, encoder1, encoder2, assembler, normalizer, featureIndexer, classmodel, labelConverter))
//val pipeline = new Pipeline().setStages(Array(sexIndex, embarkedIndex, survivedIndex, assembler, normalizer, featureIndexer, classmodel, labelConverter))
val pipeline = new Pipeline().setStages(Array(sexIndex, embarkedIndex, survivedIndex, assembler, featureIndexer, classmodel, labelConverter))


// Training
//val model = pipeline.fit(train5)

// For cross validation
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("SurvivedIndex").setPredictionCol("prediction").setMetricName("accuracy")
val paramGrid = new ParamGridBuilder().addGrid(classmodel.regParam, Array(0.5, 0.1, 0.01)).addGrid(classmodel.fitIntercept).addGrid(classmodel.elasticNetParam, Array(0.0, 0.5, 1.0)).build()
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)

// Training
val model = cv.fit(train5)


// Test data pre-processig
val avg_age = test.select(mean("Age")).first()(0).asInstanceOf[Double]
val test2 = test.na.fill(avg_age, Seq("Age"))
val test3 = test2.withColumn("Sex", addChild(test2("Sex"), test2("Age")))
val test4 = test3.withColumn("Family", withFamily(test3("SibSp"), test3("Parch")))

println("Total row count: " + test4.count)
test4.schema.map(a=>a.name).foreach{name=> test4.groupBy(test4(name)).count.sort(desc("count")).show}


val getZero = sqlContext.udf.register("toDouble", ((n: Int) => { 0 }))
val test5 = test4.withColumn("Survived", getZero(test4("PassengerId")))
val test6 = test5.na.drop()


// Test 
val result = model.transform(test6)

//result.schema.map(a=>a.name).foreach{name=> result.groupBy(result(name)).count.sort(desc("count")).show}
//result.select("PassengerId", "predictedLabel").write.format("com.databricks.spark.csv").option("header", "true").save("data_result/titanic_result.csv")


//Evaluator 
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("SurvivedIndex").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(result)
println("Test Error = " + (1.0 - accuracy))



