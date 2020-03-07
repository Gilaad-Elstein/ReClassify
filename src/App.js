import React, { Component } from 'react';
import Popup from './Popup';  
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs'
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import {
  BrowserView,
  MobileView,
  isBrowser,
  isMobile
} from "react-device-detect";

import videoLoadingImage from './img/camera.png';
import cameraFlipIcon from './img/flipCamera.png';
import './App.css';


const classifier = knnClassifier.create();

class App extends Component {


  constructor(props){
    super(props);
    this.state = {
      Prediction: "none",
      Confidence:  0,
      ModelRunning: true,
      VideoCapture: true,
      Reset: false,
      Webcam: null,
      FacingMode: 'environment',
      ShowPopup: true
    }
    this.webcamRef = React.createRef();
  }

  togglePopup() {  
    this.setState({  
         ShowPopup: !this.state.ShowPopup  
    });  
     }  


  _handleKeyDown = (event) => {
    var ESCAPE_KEY = 27;
      switch( event.keyCode ) {
          case ESCAPE_KEY:
              this.toggleVideoCapture();
              break;
          default: 
              break;
      }
  }

  toggleCameraMode = () => {
    if (isBrowser){
      return;
    }
    if(this.state.FacingMode === 'environment'){
      this.setState({FacingMode: 'user'}, ()=>{
        this.state.Webcam.stop();
        this.maybeLoadWebcam(true);
      });
    }
    else{
      this.setState({FacingMode: 'environment'}, ()=>{
        this.state.Webcam.stop();
        this.maybeLoadWebcam(true);
      });
    }
  }

  async maybeLoadWebcam(force){
    if (this.state.Webcam == null || force){
      // Create an object from Tensorflow.js data API which could capture image 
      // from the web camera as Tensor.
      const webcam = await tf.data.webcam(this.webcamRef.current, {facingMode: this.state.FacingMode});
      this.setState({Webcam: webcam});
      }
      return this.state.Webcam;
    }

  async toggleVideoCapture(){
  if(this.state.VideoCapture && this.state.Webcam !== null){
    this.state.Webcam.stop();
    this.setState({VideoCapture: false});
  }
  else{
    const webcam = await tf.data.webcam(this.webcamRef.current);
    this.setState({Webcam: webcam});
    this.setState({VideoCapture: true});
  }
}

  componentDidMount(){
    document.addEventListener("keydown", this._handleKeyDown);

    if(typeof InstallTrigger !== 'undefined'){
      alert("This demo does not currently support firefox.\r\nWe recommended running it in Chrome instead.");
      this.setState({Prediction: "This demo  does not currently support firefox."});
    }
    else{
      this.runModel();
    }
  }

  async runModel() {
    const net = await mobilenet.load();
    if(this.state.Webcam == null){
      await this.maybeLoadWebcam();
    }
    // Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = async classId => {
      if (this.state.VideoCapture){
    // Capture an image from the web camera.
    const img = await this.state.Webcam.capture();
      
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(img, 'conv_preds');

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    img.dispose();
      }
    };
  
    // When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));
  
    while (this.state.ModelRunning) {
      if (this.state.Reset){
        classifier.clearAllClasses();
        this.setState({Confidence: 0, Prediction: "none", Reset: false});
      }
      if (classifier.getNumClasses() > 0 && this.state.VideoCapture) {
        const img = await this.state.Webcam.capture();
        
        // Get the activation from mobilenet from the webcam.
        var activation;
        try{
          activation = net.infer(img, 'conv_preds');
        }
        catch{
          setTimeout(() => {
            this.runModel();
          }, 1000);
          return;
        }
        // Get the most likely class and confidence from the classifier module.
        const result = await classifier.predictClass(activation);
  
        const classes = ['A', 'B', 'C'];
        //console.log("prediction: " + classes[result.label] + "\n probability: " + result.confidences[result.label]);
        this.setState({Prediction: classes[result.label], Confidence: result.confidences[result.label]})
        // Dispose the tensor to release the memory.
        img.dispose();
      }
  
      await tf.nextFrame();
    }
    this.setState({Prediction: "Model suspended.", Confidence: 0});
  }

  getConfidence(){
    return "Confidence: " +  (this.state.ModelRunning ? 
            (this.state.Confidence * 100).toFixed(0) + "%" : " --- ");
  }

  getPrediction(){
    return this.state.ModelRunning ? 
    "Prediction: " + (this.state.Prediction) : "Model suspended";
  }
 
  requestReset = () => {
    this.setState({Reset: true});
  }

  render() {
    return(
    <div className="App">
      
      <header className="App-header">
      {this.state.ShowPopup ?  
        <Popup  
          header='ReClassify'
          text="Reclassify is an on the fly image classifier.~n

          It can learn to tell the difference between facial
          expressions such as a smile or a frown, it can tell
          genders apart, differentiate hand gestures or classify
          inanimate objects with as little as 5-10 training examples per class.~n~n
         
          Point your camera at an object or scene and
          click one of the class buttons.~n~n
          
          Every time you click a class button, ReClassify learns to associate
          what it sees with that class.~n~n
          
          Try taking different perspectives on the object
          to maximize the model's confidence.~n~n
         
          Repeat with a different object or scene and 
          a different class button.~n~n
          
          When you are done simply point your camera at an
          object of one of the classes and see the model's
          prediction. The more you click, the better it predicts!~n~n
          
          Enjoy!"
          closePopup={this.togglePopup.bind(this)}  
        />  
        : null  
        } 
      <h1>ReClassify</h1>
      <h4 className="h4">On the fly image classification</h4>

      </header>
      <div className="App-main">

      <video ref={this.webcamRef}
        autoPlay
        playsInline 
        muted 
        id="webcam"
        poster={videoLoadingImage}
        width="224" 
        height="224">
      </video>

      <MobileView>
      <div><input className="label" type="image" alt="&#8635;" width={50} height={50} id="toggleCameraMode" onClick={this.toggleCameraMode} src={cameraFlipIcon}></input></div>
        </MobileView>

      <label className="label" htmlFor="prediction">{this.state.Prediction}<br></br>{this.getConfidence()}</label>
      <div className="buttons">
      <button id="class-a" >Class A</button>&nbsp;&nbsp;
      <button id="class-b">Class B</button>&nbsp;&nbsp;
      <button id="class-c">Class C</button></div>
      <br></br>
      <button id="reset" onClick={this.requestReset}>Reset model</button>  

      </div>
    </div>
  );
}
}

export default App;
