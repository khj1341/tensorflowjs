import React, { useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as dfd from "danfojs";

const Classification = () => {
  const csvUrl =
    "https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv";

  const classifyHandler = async () => {
    const data = await dfd.readCSV(csvUrl);

    const 독립변수 = data.loc({
      columns: ["꽃잎길이", "꽃잎폭", "꽃받침길이", "꽃받침폭"],
    }).tensor;
    // 독립변수.print();

    // 종속변수가 숫자가 아니기 떄문에 숫자로 변환해줌.
    const encoder = new dfd.OneHotEncoder();
    const 종속변수 = tf.tensor(
      encoder.fit(data["품종"]).transform(data["품종"].values)
    );
    // data["품종"].print();
    종속변수.print();

    const X = tf.input({ shape: [4] });
    const H = tf.layers.dense({ units: 4, activation: "relu" }).apply(X);
    // const Y = tf.layers.dense({ units: 3 }).apply(H);
    // softmax: 0 ~ 1 사이의 값만 나오도록 해줌.
    const Y = tf.layers.dense({ units: 3, activation: "softmax" }).apply(H); // OneHotEncoding을 통해서 setosa, versicolor, virginica 3개의 컬럼이 있어서 output 3가지

    // @ts-ignore
    const model = tf.model({ inputs: X, outputs: Y });

    // 회귀에서 사용
    // const compileParam = {
    //   optimizer: tf.train.adam(),
    //   loss: tf.losses.meanSquaredError,
    // };

    // 분류에서 사용
    const compileParam = {
      optimizer: tf.train.adam(),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"], // 정확도라는 지표도 사용할 수 있음. => tfvis.show.history  acc 에서 볼 수 있음.
    };

    model.compile(compileParam);

    tfvis.show.modelSummary({ name: "요약", tab: "모델" }, model);

    const _history: tf.ModelFitArgs[] = [];
    const fitParam = {
      epochs: 500,
      callbacks: {
        onEpochEnd: function (
          epoch: number,
          logs: tf.ModelFitArgs | undefined
        ) {
          // @ts-ignore
          console.log("epoch", epoch, logs, "RMSE=>", Math.sqrt(logs.loss));
          if (logs) {
            _history.push(logs);
          }
          // @ts-ignore
          tfvis.show.history({ name: "loss", tab: "역사" }, _history, ["loss"]);
          // @ts-ignore
          tfvis.show.history({ name: "accuracy", tab: "역사" }, _history, [
            "acc",
          ]);
        },
      },
    };

    // @ts-ignore
    model.fit(독립변수, 종속변수, fitParam).then(function (result) {
      // 4. 모델을 이용합니다.
      // 4.1 기존의 데이터를 이용
      const 예측한결과 = new dfd.DataFrame(model.predict(독립변수));
      예측한결과.print();
      종속변수.print();
    });
  };

  useEffect(() => {
    classifyHandler();
  }, []);

  return <div>Classification</div>;
};

export default Classification;
