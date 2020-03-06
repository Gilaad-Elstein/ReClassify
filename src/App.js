import React, { Component } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs'
import * as knnClassifier from '@tensorflow-models/knn-classifier';


import './App.css';


let net;
const classifier = knnClassifier.create();

class App extends Component {


  constructor(props){
    super(props);
    this.state = {
      Prediction: "none",
      Confidence:  0,
      Running: true,
      Reset: false
    }
    this.webcam = React.createRef();
  }

  componentDidMount(){
    this.app();
  }

  async app() {
    console.log('Loading mobilenet..');
  
    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');
  
    // Create an object from Tensorflow.js data API which could capture image 
    // from the web camera as Tensor.
    const webcam = await tf.data.webcam(this.webcam.current);
  
    // Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = async classId => {
      // Capture an image from the web camera.
      const img = await webcam.capture();
  
      // Get the intermediate activation of MobileNet 'conv_preds' and pass that
      // to the KNN classifier.
      const activation = net.infer(img, 'conv_preds');
  
      // Pass the intermediate activation to the classifier.
      classifier.addExample(activation, classId);
  
      // Dispose the tensor to release the memory.
      img.dispose();
    };
  
    // When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));
  
    while (this.state.Running) {
      if (this.state.Reset){
        classifier.clearAllClasses();
        this.setState({Confidence: 0, Prediction: "none", Reset: false});
      }
      if (classifier.getNumClasses() > 0) {
        const img = await webcam.capture();
  
        // Get the activation from mobilenet from the webcam.
        const activation = net.infer(img, 'conv_preds');
        // Get the most likely class and confidence from the classifier module.
        const result = await classifier.predictClass(activation);
  
        const classes = ['A', 'B', 'C'];
        console.log("prediction: " + classes[result.label] + "\n probability: " + result.confidences[result.label]);
        this.setState({Prediction: classes[result.label], Confidence: result.confidences[result.label]})
        // Dispose the tensor to release the memory.
        img.dispose();
      }
  
      await tf.nextFrame();
    }
    this.setState({Prediction: "Model suspended.", Confidence: 0});
  }

  toggleRunning = () => {
    this.setState({Running: !this.state.Running});
    if(!this.state.Running){
      this.setState({Prediction: "none"});
      this.app();
    }
  }

  getConfidence(){
    return "Confidence: " +  (this.state.Running ? 
            (this.state.Confidence * 100).toFixed(0) + "%" : " --- ");
  }

  getPrediction(){
    return this.state.Running ? 
    "Prediction: " + (this.state.Prediction) : "Model suspended";
  }

  requestReset = () => {
    this.setState({Reset: true});
  }

  render() {
    return(
    <div className="App">
      
      <header className="App-header">
      
      <h1>ReClassify</h1>
      <h4 className="h4">On the fly image classification</h4>

      </header>
      <div className="App-main">
      <video ref={this.webcam} autoPlay playsInline muted id="webcam" width="224" height="224"></video>
      <label className="label" htmlFor="prediction">{this.state.Prediction}<br></br>{this.getConfidence()}</label>
      <div className="buttons">
      <button id="class-a" >Class A</button>&nbsp;&nbsp;
      <button id="class-b">Class B</button>&nbsp;&nbsp;
      <button id="class-c">Class C</button></div>
      <br></br>
    <button id="toggleRunning" onClick={this.toggleRunning}>Start/Stop</button>&nbsp;&nbsp;
    <button id="reset" onClick={this.requestReset}>Reset model</button>

      </div>
    </div>

    
  );
}
}

export default App;
