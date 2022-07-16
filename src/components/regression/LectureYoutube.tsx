import React, { useEffect } from "react";
import * as mobilenet from "@tensorflow-models/mobilenet";

import dogImage from "../assets/dog.jpeg";

const LectureYoutube = () => {
  const runMobilenet = async () => {
    const img2 = new Image(30, 30);
    img2.src = dogImage;
    // Load the model.
    try {
      const model = await mobilenet.load();
      // Classify the image.
      const predictions = await model.classify(img2);

      console.log("Predictions: ");
      console.log(predictions);
    } catch (err) {
      console.log({ err });
    }
  };

  useEffect(() => {
    runMobilenet();
    console.log("test");
  }, []);

  return <div>LectureYoutube</div>;
};

export default LectureYoutube;
