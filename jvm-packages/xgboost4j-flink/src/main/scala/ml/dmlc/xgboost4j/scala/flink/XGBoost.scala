/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.flink

import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.java.{Rabit, RabitTracker}
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => XGBoostScala}
import org.apache.commons.logging.LogFactory
import org.apache.flink.api.common.functions.RichMapPartitionFunction
import org.apache.flink.api.java.DataSet
import org.apache.flink.util.Collector
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

import scala.jdk.CollectionConverters._

object XGBoost {
  /**
    * Helper map function to start the job.
    *
    * @param workerEnvs
    */
  private class MapFunction(paramMap: Map[String, Any],
                            round: Int,
                            workerEnvs: java.util.Map[String, String])
    extends RichMapPartitionFunction[(Float, Vector[Float]), XGBoostModel] {
    val logger = LogFactory.getLog(this.getClass)

    def mapPartition(it: java.lang.Iterable[(Float, Vector[Float])],
                     collector: Collector[XGBoostModel]): Unit = {
      workerEnvs.put("DMLC_TASK_ID", String.valueOf(this.getRuntimeContext.getIndexOfThisSubtask))
      logger.info("start with env" + workerEnvs.toString)
      Rabit.init(workerEnvs)
      val mapper = (x: (Float, Vector[Float])) => {
        val vector = x._2.map(_.toFloat)
        LabeledPoint(x._1.toFloat, vector.size, vector.indices.toArray, vector.toArray)
      }
      val dataIter = for (x <- it.iterator().asScala) yield mapper(x)
      val trainMat = new DMatrix(dataIter, null)
      val watches = List("train" -> trainMat).toMap
      val round = 2
      val numEarlyStoppingRounds = paramMap.get("numEarlyStoppingRounds")
          .map(_.toString.toInt).getOrElse(0)
      val booster = XGBoostScala.train(trainMat, paramMap, round, watches,
        earlyStoppingRound = numEarlyStoppingRounds)
      Rabit.shutdown()
      collector.collect(new XGBoostModel(booster))
    }
  }

  /**
    * Load XGBoost model from path, using Hadoop Filesystem API.
    *
    * @param modelPath The path that is accessible by hadoop filesystem API.
    * @return The loaded model
    */
  def loadModelFromHadoopFile(modelPath: String) : XGBoostModel = {
    new XGBoostModel(
      XGBoostScala.loadModel(FileSystem.get(new Configuration).open(new Path(modelPath))))
  }

  /**
    * Train a xgboost model with link.
    *
    * @param dtrain The training data.
    * @param params The parameters to XGBoost.
    * @param round Number of rounds to train.
    */
  def train(dtrain: DataSet[(Float, Vector[Float])], params: Map[String, Any], round: Int):
      XGBoostModel = {
    val tracker = new RabitTracker(dtrain.getExecutionEnvironment.getParallelism)
    if (tracker.start(0L)) {
      dtrain
        .mapPartition(new MapFunction(params, round, tracker.getWorkerEnvs))
        .reduce((x, _) => x).collect().get(0)
    } else {
      throw new Error("Tracker cannot be started")
      null
    }
  }
}
