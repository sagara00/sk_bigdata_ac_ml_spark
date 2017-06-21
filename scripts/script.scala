
/************************************ MLlib Pacakge 실습 ******************************************/

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors



// covtype
//val data2 = MLUtils.loadLibSVMFile(sc, "/Users/freeman/programming/spark/data/mllib/covtype.libsvm.binary.scale")
//val data = data2.map{lp => var labelp = None: Option[LabeledPoint]; if (lp.label == 1.0) labelp = Some(LabeledPoint(0.0, lp.features))	else if (lp.label == 2.0) labelp = Some(LabeledPoint(1.0, lp.features)); labelp.get }


// breast-cancer
val data2 = MLUtils.loadLibSVMFile(sc, "data/mllib/breast-cancer_scale")
val data = data2.map{lp => var labelp = None: Option[LabeledPoint]; if (lp.label == 2.0) labelp = Some(LabeledPoint(0.0, lp.features))	else if (lp.label == 4.0) labelp = Some(LabeledPoint(1.0, lp.features)); labelp.get }


//val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")


// Adult (a9a)
val data2 = MLUtils.loadLibSVMFile(sc, "data/mllib/a9a")
val data = data2.map{lp => var labelp = None: Option[LabeledPoint]; if (lp.label == -1.0) labelp = Some(LabeledPoint(0.0, lp.features))	else if (lp.label == +1.0) labelp = Some(LabeledPoint(1.0, lp.features)); labelp.get }


// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)
val training = splits(0).cache()
val test = splits(1)


// SVM
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
val svmAlg = new SVMWithSGD()
//svmAlg.optimizer.setNumIterations(200).setRegParam(0.1).setUpdater(new L1Updater)
svmAlg.optimizer.setNumIterations(100).setRegParam(1.0)
val model = svmAlg.run(training)


// Logistic Regression
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)


// Decision Tree
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 5
val maxBins = 32
val model = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)


// Naive Bayes
//import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
//val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")


// Compute raw scores on the test set.
val scoreAndLabels = test.map { point => 
val score = model.predict(point.features)
 (score, point.label)
}

// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()

println("Area under ROC = " + auROC)


// Get evaluation metrics.
val metrics = new MulticlassMetrics(scoreAndLabels)
val precision = metrics.precision
println("Precision = " + precision)


/************************************ DataFrame  실습 ******************************************/

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row

val training = spark.createDataFrame(Seq(
  (1.0, Vectors.dense(0.0, 1.1, 0.1)),
  (0.0, Vectors.dense(2.0, 1.0, -1.0)),
  (0.0, Vectors.dense(2.0, 1.3, 1.0)),
  (1.0, Vectors.dense(0.0, 1.2, -0.5))
)).toDF("label", "features")
training.count
training.show


val df = spark.read.json("examples/src/main/resources/people.json")
df.count
df.show

// val adult_df = spark.read.json()
// df.count
// df.show


// spark.csv library 이용
import org.apache.spark.sql.SQLContext
import spark.implicits._
val sqlContext = new SQLContext(sc)
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "false").option("inferSchema", "true").load("data/mllib/adult.data")
df.show
df.printSchema

// rename column names
val df2 = df.withColumnRenamed("_c0", "age").withColumnRenamed("_c1","workclass").withColumnRenamed("_c2", "fnlwgt").withColumnRenamed("_c3","education").withColumnRenamed("_c4","education-num").withColumnRenamed("_c5","marital-status").withColumnRenamed("_c6","occupation").withColumnRenamed("_c7","relationship").withColumnRenamed("_c8","race").withColumnRenamed("_c9","sex").withColumnRenamed("_c10","capital-gain").withColumnRenamed("_c11","capital-loss").withColumnRenamed("_c12","hours-per-week").withColumnRenamed("_c13","native-country").withColumnRenamed("_c14","label")
df2.show
df2.printSchema


// select columns
val select_df = df2.select("education", "occupation")
select_df.show

// filter
val filter_df = df2.filter($"age" > 50)
filter_df.count
filter_df.show

// groupBy
val groupBy_df = df2.groupBy("marital-status").count
groupBy_df.show


// groupBy 후 내림차순 sorting
val groupBySort_df = df2.groupBy("marital-status").count.sort(desc("count"))
groupBySort_df.show


// null 혹은 empty space 찾기
df2.filter(df2("race").isNull || df2("race") === "").count
 


/************************************ Spark ML package 실습 ******************************************/

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row

// Prepare training data from a list of (label, features) tuples.
val training = spark.createDataFrame(Seq(
  (1.0, Vectors.dense(0.0, 1.1, 0.1)),
  (0.0, Vectors.dense(2.0, 1.0, -1.0)),
  (0.0, Vectors.dense(2.0, 1.3, 1.0)),
  (1.0, Vectors.dense(0.0, 1.2, -0.5))
)).toDF("label", "features")


// Create a LogisticRegression instance. This instance is an Estimator.
val lr = new LogisticRegression()
// Print out the parameters, documentation, and any default values.
println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")


// We may set parameters using setter methods.
lr.setMaxIter(10).setRegParam(0.01)


// Learn a LogisticRegression model. This uses the parameters stored in lr.
val model1 = lr.fit(training)
// Since model1 is a Model (i.e., a Transformer produced by an Estimator),
// we can view the parameters it used during fit().
// This prints the parameter (name: value) pairs, where names are unique IDs for this
// LogisticRegression instance.
println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)

// We may alternatively specify parameters using a ParamMap,
// which supports several methods for specifying parameters.
val paramMap = ParamMap(lr.maxIter -> 20).put(lr.maxIter, 30).put(lr.regParam -> 0.1, lr.threshold -> 0.55)

// One can also combine ParamMaps.
val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")  // Change output column name.
val paramMapCombined = paramMap ++ paramMap2

// Now learn a new model using the paramMapCombined parameters.
// paramMapCombined overrides all parameters set earlier via lr.set* methods.
val model2 = lr.fit(training, paramMapCombined)
println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)

// Prepare test data.
val test = spark.createDataFrame(Seq(
  (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
  (0.0, Vectors.dense(3.0, 2.0, -0.1)),
  (1.0, Vectors.dense(0.0, 2.2, -1.5))
)).toDF("label", "features")


// Make predictions on test data using the Transformer.transform() method.
// LogisticRegression.transform will only use the 'features' column.
// Note that model2.transform() outputs a 'myProbability' column instead of the usual
// 'probability' column since we renamed the lr.probabilityCol parameter previously.
model2.transform(test).select("features", "label", "myProbability", "prediction").collect().foreach {case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>println(s"($features, $label) -> prob=$prob, prediction=$prediction")}



// **************  Feature Extractor - Word2Vec

import org.apache.spark.ml.feature.Word2Vec

// Input data: Each row is a bag of words from a sentence or document.
val documentDF = spark.createDataFrame(Seq(
  "Hi I heard about Spark".split(" "),
  "I wish Java could use case classes".split(" "),
  "Logistic regression models are neat".split(" ")
).map(Tuple1.apply)).toDF("text")


// Learn a mapping from words to Vectors.
val word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(3).setMinCount(0)
val model = word2Vec.fit(documentDF)
val result = model.transform(documentDF)
result.select("result").take(4).foreach(println)


// **************  Feature Transformer - Tokenizer

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}

val sentenceDataFrame = spark.createDataFrame(Seq(
  (0, "Hi I heard about Spark"),
  (1, "I wish Java could use case classes"),
  (2, "Logistic,regression,models,are,neat")
)).toDF("label", "sentence")

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")


val tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.select("words", "label").take(3).foreach(println)



// **************  Feature Transformer - PCA

import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors

val data = Array(
  Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
  Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
  Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
)

val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(3).fit(df)
val pcaDF = pca.transform(df)
val result = pcaDF.select("pcaFeatures")
result.foreach(println(_))


// **************  Feature Transformer - StringIndexer

import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(
  Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
).toDF("id", "category")

val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)
indexed.show()


// **************  Feature Transformer - IndexToString


import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

val df = spark.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
)).toDF("id", "category")

val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex").fit(df)
val indexed = indexer.transform(df)

indexed.show
val converter = new IndexToString().setInputCol("categoryIndex").setOutputCol("originalCategory")

val converted = converter.transform(indexed)
converted.select("id", "originalCategory").show()


// **************  Feature Transformer - VectorIndexer (Example)


import org.apache.spark.ml.feature.VectorIndexer

val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val indexer = new VectorIndexer().setInputCol("features").setOutputCol("indexed").setMaxCategories(10)

val indexerModel = indexer.fit(data)

val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
println(s"Chose ${categoricalFeatures.size} categorical features: " + categoricalFeatures.mkString(", "))

// Create new column "indexed" with categorical values transformed to indices
val indexedData = indexerModel.transform(data)
indexedData.show()



// **************  Feature Transformer - VectorIndexer

import org.apache.spark.ml.linalg.Vectors

val data = Array(
  Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0),
  Vectors.dense(1.0, 1.0, 2.0, 3.0, 4.0),
  Vectors.dense(1.0, 2.0, 1.0, 2.0, 3.0),
  Vectors.dense(1.0, 1.0, 3.0, 1.0, 2.0),
  Vectors.dense(1.0, 1.0, 2.0, 1.0, 1.0)
)

val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

val indexer = new VectorIndexer().setInputCol("features").setOutputCol("indexed").setMaxCategories(3)
val indexerModel = indexer.fit(df)

// Create new column "indexed" with categorical values transformed to indices
val indexedData = indexerModel.transform(df)
indexedData.take(5).foreach(println(_))



// **************  Feature Transformer - Normalizer

import org.apache.spark.ml.feature.Normalizer

val data = Array(
  Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0),
  Vectors.dense(1.0, 1.0, 2.0, 3.0, 4.0),
  Vectors.dense(1.0, 2.0, 1.0, 2.0, 3.0),
  Vectors.dense(1.0, 1.0, 3.0, 1.0, 2.0),
  Vectors.dense(1.0, 1.0, 2.0, 1.0, 1.0)
)
val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

// Normalize each Vector using $L^1$ norm.
val normalizer = new Normalizer().setInputCol("features").setOutputCol("normFeatures").setP(2.0)

val l1NormData = normalizer.transform(df)
l1NormData.take(5).foreach(println(_))



// **************  Feature Transformer - Vector Assembler

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val dataset = spark.createDataFrame(
  Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
).toDF("id", "hour", "mobile", "userFeatures", "clicked")

val assembler = new VectorAssembler().setInputCols(Array("hour", "mobile", "userFeatures")).setOutputCol("features")

val output = assembler.transform(dataset)
println(output.select("features").first())


// **************  Feature Transformer - QuantileDiscretizer


import org.apache.spark.ml.feature.QuantileDiscretizer

val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
var df = spark.createDataFrame(data).toDF("id", "hour")

val discretizer = new QuantileDiscretizer().setInputCol("hour").setOutputCol("result").setNumBuckets(3)

val result = discretizer.fit(df).transform(df)
result.show()



// **************  Pipeline 


import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

// Prepare training documents from a list of (id, text, label) tuples.
val training = spark.createDataFrame(Seq(
  (0L, "a b c d e spark", 1.0),
  (1L, "b d", 0.0),
  (2L, "spark f g h", 1.0),
  (3L, "hadoop mapreduce", 0.0)
)).toDF("id", "text", "label")

// Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01)
val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

// Fit the pipeline to training documents.
val model = pipeline.fit(training)

// Prepare test documents, which are unlabeled (id, text) tuples.
val test = spark.createDataFrame(Seq(
  (4L, "spark i j k"),
  (5L, "l m n"),
  (6L, "mapreduce spark"),
  (7L, "apache hadoop")
)).toDF("id", "text")

// Make predictions on test documents.
model.transform(test).select("id", "text", "probability", "prediction").collect().foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
    println(s"($id, $text) --> prob=$prob, prediction=$prediction")
}


// Now we can optionally save the fitted pipeline to disk
model.write.overwrite().save("data/mllib_save/spark-logistic-regression-model")

// We can also save this unfit pipeline to disk
pipeline.write.overwrite().save("data/mllib_save/unfit-lr-model")

// And load it back in during production
val sameModel = PipelineModel.load("data/mllib_save/spark-logistic-regression-model")



// **************  Classification: Logistic Regression

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Load training data
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val Array(training, test) = data.randomSplit(Array(0.7, 0.3))

val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

// Fit the model
val lrModel = lr.fit(training)

// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

val predictions = lrModel.transform(test)
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))


import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}

// Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
// example
val trainingSummary = lrModel.summary

// Obtain the objective per iteration.
val objectiveHistory = trainingSummary.objectiveHistory
objectiveHistory.foreach(loss => println(loss))


// **************  Classification: Decision Tree


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))


// Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)



// **************  Classification: Random Forest

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load and parse the data file, converting it to a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))


// Train a RandomForest model.
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)



// **************  Classification: Gradient Boosted Tree

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load and parse the data file, converting it to a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a GBT model.
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and GBT in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println("Learned classification GBT model:\n" + gbtModel.toDebugString)



// **************  Classification: Naive Bayes

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a NaiveBayes model.
val model = new NaiveBayes().fit(trainingData)

// Select example rows to display.
val predictions = model.transform(testData)
predictions.show()

// Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Accuracy: " + accuracy)


// **************  Regression: Linear Regressoion

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

// Load training data
val data = spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")

val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

// Split the data into training and test sets (30% held out for testing)
val Array(training, test) = data.randomSplit(Array(0.7, 0.3))

// Fit the model
val lrModel = lr.fit(training)

val predictions = lrModel.transform(test)
predictions.show()


// Select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)


// Print the coefficients and intercept for linear regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarize the model over the training set and print out some metrics
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")


// **************  Clustering: K-means

import org.apache.spark.ml.clustering.KMeans

// Loads data.
val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

// Trains a k-means model.
val kmeans = new KMeans().setK(2).setSeed(1L)
val model = kmeans.fit(dataset)

// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(dataset)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

val presult = model.transform(dataset)
presult.show





