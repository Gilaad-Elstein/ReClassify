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
      Running: true
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
      if (classifier.getNumClasses() > 0) {
        const img = await webcam.capture();
  
        // Get the activation from mobilenet from the webcam.
        const activation = net.infer(img, 'conv_preds');
        // Get the most likely class and confidence from the classifier module.
        const result = await classifier.predictClass(activation);
  
        const classes = ['A', 'B', 'C'];
        console.log("prediction: " + classes[result.label] + "\n probability: " + result.confidences[result.label]);
        this.setState({Prediction: classes[result.label]})
        // Dispose the tensor to release the memory.
        img.dispose();
      }
  
      await tf.nextFrame();
    }
  }

  toggleRunning = () => {
    this.setState({Running: !this.state.Running});
    if(!this.state.Running){
      this.app();
    }
  }

  isDevReact() {
    try {
      React.createClass({});
    } catch(e) {
      if (e.message.indexOf('render') >= 0) {
        return true;  // A nice, specific error message
      } else {
        return false;  // A generic error message
      }
    }
    return false;  // should never happen, but play it safe.
  };

  render() {
    return(
    <div className="App">
      
      <header className="App-header">
      
      <h1>ReClassify</h1>

      </header>
      <div className="App-main">
      <video ref={this.webcam} autoPlay playsInline muted id="webcam" width="224" height="224"></video>
      <button id="class-a">Add A</button>
      <button id="class-b">Add B</button>
      <button id="class-c">Add C</button>
    <label htmlFor="prediction">{this.state.Prediction}</label>
    <button id="toggleRunning" onClick={this.toggleRunning}>Start/Stop</button>
      </div>
    </div>

    
  );
}
}

export default App;
