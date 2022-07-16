import React, { useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

const Lecture5 = () => {
  const runML = async () => {
    // 1. 과거의 데이터를 준비합니다.
    const 온도 = [20, 21, 22, 23];
    const 판매량 = [40, 42, 44, 46];
    const 원인 = tf.tensor(온도);
    const 결과 = tf.tensor(판매량);
    // 원인.print();
    // 결과.print();

    // 2. 모델의 모양을 만듭니다.
    try {
      const X = tf.input({ shape: [1] });
      const Y = tf.layers.dense({ units: 1 }).apply(X);
      // @ts-ignore
      const model = tf.model({ inputs: X, outputs: Y });
      const compileParam = {
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
      };
      model.compile(compileParam);

      // 3. 데이터로 모델을 학습시킵니다.
      // const fitParam = { epochs: 10000 };
      const fitParam = {
        epochs: 10000,
        callbacks: {
          onEpochEnd: function (
            epoch: number,
            logs: tf.ModelFitArgs | undefined
          ) {
            // @ts-ignore
            console.log("epoch", epoch, logs, "RMSE => ", Math.sqrt(logs.loss));
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

      // 4.2 새로운 데이터를 이용
      const 다음주온도 = [15, 16, 17, 18, 19];
      const 다음주원인 = tf.tensor(다음주온도);
      const 다음주결과 = model.predict(다음주원인);
      // @ts-ignore
      console.log({ test: 다음주결과.arraySync() }); // Tensor => Array

      const weights = model.getWeights();
      // @ts-ignore
      const weight = weights[0].arraySync()[0][0];
      // @ts-ignore
      const bias = weights[1].arraySync()[0];
      console.log({ weight });
      console.log({ bias });

      // @ts-ignore
      // 다음주결과.print();

      // await model.save("downloads://lemon");
      await model.save("localstorage://lemon");
    } catch (error) {}
  };

  useEffect(() => {
    runML();
  }, []);

  return <div>5</div>;
};

export default Lecture5;
