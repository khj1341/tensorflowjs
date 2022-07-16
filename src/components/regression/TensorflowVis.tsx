import React, { useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import { 보스톤_원인, 보스톤_결과 } from "./constants";

const TensorflowVis = () => {
  const runML = async () => {
    // 1. 과거의 데이터를 준비합니다.
    const 원인 = tf.tensor(보스톤_원인);
    const 결과 = tf.tensor(보스톤_결과);
    // 원인.print();
    // 결과.print();

    // 2. 모델의 모양을 만듭니다.
    try {
      const X = tf.input({ shape: [13] });
      const Y = tf.layers.dense({ units: 1 }).apply(X);
      // @ts-ignore
      const model = tf.model({ inputs: X, outputs: Y });
      const compileParam = {
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
      };
      model.compile(compileParam);

      const surfaceSummary = {
        name: "요약",
        tab: "모델",
      };
      tfvis.show.modelSummary(surfaceSummary, model);

      // 3. 데이터로 모델을 학습시킵니다.

      let _history: tf.ModelFitArgs[] = [];
      // const fitParam = { epochs: 10000 };
      const fitParam = {
        epochs: 100,
        callbacks: {
          onEpochEnd: function (
            epoch: number,
            logs: tf.ModelFitArgs | undefined
          ) {
            // @ts-ignore
            console.log("epoch", epoch, logs, "RMSE => ", Math.sqrt(logs.loss));
            if (logs) {
              _history.push(logs);
            }
            const surfaceHistory = {
              name: "loss",
              tab: "역사",
            };
            // @ts-ignore
            tfvis.show.history(surfaceHistory, _history, ["loss"]);
          },
        },
      };
      // loss 추가 예제
      await model.fit(원인, 결과, fitParam);
      // 4. 모델을 이용합니다.
      // 4.1 기존의 데이터를 이용
      const 예측한결과 = model.predict(원인);
      // @ts-ignore
      예측한결과.print();

      const weights = model.getWeights();
      const weight = weights[0].arraySync();
      const bias = weights[1].arraySync();
      console.log({ weight, bias });
    } catch (error) {}
  };

  useEffect(() => {
    runML();
  }, []);

  return <div>TensorflowVis</div>;
};

export default TensorflowVis;
