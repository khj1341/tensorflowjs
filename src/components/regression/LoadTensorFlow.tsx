import React, { useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

const LoadTensorFlow = () => {
  const loadModel = async () => {
    const model = await tf.loadLayersModel("localstorage://lemon");
    // @ts-ignore
    model.predict(tf.tensor([20])).print();
  };

  useEffect(() => {
    loadModel();
  }, []);
  return <div>LoadTensorFlow</div>;
};

export default LoadTensorFlow;
