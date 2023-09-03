/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const CAT_NUM = [393, 491, 400, 584, 472, 127];
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];


// Just add more buttons in HTML to allow classification of more classes of data!

let mobilenet = undefined;
let onclass = 0;
let onnum = 11;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];

document.getElementById("train").addEventListener('click', function(event) {
  predict("./data/test/cardboard/cardboard1.jpg");
});

/**
 * Loads the MobileNet model and warms it up so ready for use.
 **/

function loadImage(src) {  
  return new Promise((resolve, reject) => {    
      const img = new Image();
      img.src = src;    
      img.onload = () => resolve(tf.browser.fromPixels(img));    
      img.onerror = (err) => reject(err);  
  });
}

function cropImage(img) {  
  const width = img.shape[0];  
  const height = img.shape[1];
  const shorterSide = Math.min(img.shape[0], img.shape[1]);
  const startingHeight = (height - shorterSide) / 2;
  const startingWidth = (width - shorterSide) / 2;
  const endingHeight = startingHeight + shorterSide;
  const endingWidth = startingWidth + shorterSide;
  return img.slice([startingWidth, startingHeight, 0], [endingWidth, endingHeight, 3]);
}

function resizeImage(image) {
  return tf.image.resizeBilinear(image, [224, 224]);
}

function batchImage(image) {
  const batchedImage = image.expandDims(0);  
  return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
}

function loadAndProcessImage(image) {  
  const croppedImage = cropImage(image);  
  const resizedImage = resizeImage(croppedImage);  
  const batchedImage = batchImage(resizedImage);  
  return batchedImage;
}

let model = undefined;


/**
 * Check if getUserMedia is supported for webcam access.
 **/

/**
 * Enable the webcam with video constraints applied.
 **/


/**
 * Handle Data Gather for button mouseup/mousedown.
 **/


/**
 *  Make live predictions from webcam once trained.
 **/
function predict(src) {
  tf.tidy(function() {
    loadImage(src).then(img => {    
      const processedImage = loadAndProcessImage(img);    
      const prediction = model.predict(processedImage);     
      prediction.print(); 
      let highestIndex = prediction.argMax().arraySync();
      let predictionArray = prediction.arraySync();
      console.log('Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence');
    });
  });
}

async function loadMobileNetFeatureModel() {
  const URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
  model = await tf.loadLayersModel('https://eco-scan.github.io/ml-classifier-cardboard-glass-metal-paper-plastic-trash.json');
  
  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer.shape);
  });
}

loadMobileNetFeatureModel();
/**
 * Purge data and start over. Note this does not dispose of the loaded 
 * MobileNet model and MLP head tensors as you will need to reuse 
 * them to train a new model.
 **/
