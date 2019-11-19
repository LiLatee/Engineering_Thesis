import React, { Component } from 'react';
import './App.css';
import Header from './components/Header.js'
import Models from './components/Models'

class App extends Component {

  state = {
    models: []
  }

  ws_data_provider = new WebSocket("ws://0.0.0.0:8765");
  ws_evaluation_server = new WebSocket("ws://0.0.0.0:8766");

  componentDidMount() {
    this.ws_data_provider.onopen = () => {
    console.log('WS to data_provider: connected.')
    }

    this.ws_evaluation_server.onopen = () => {
    console.log('WS to evaluation_server: connected.')
    }

    this.ws_evaluation_server.onmessage = evt => {
    const message = JSON.parse(evt.data)
    // console.log(evt.data);

    var new_state = Array([]);
    for(var key in message) {
      if(message.hasOwnProperty(key)) {  
          var model = message[key];
          // for(var key2 in model) {
          //   console.log(key2);
          //   if(key2 == 'x') {
          //     console.log(key2 + " -> " + model[key2]);
          //   }
          // }
          // console.log(model);
          // var correct_prediction_ratio = model["correct_predictions"] / model["processed_samples"];
          // console.log('correct_prediction_ratio = ' + correct_prediction_ratio);
          // console.log("message[key].hasOwnProperty('x')=" + message[key].hasOwnProperty('x'));
          // if(message[key].hasOwnProperty('x')) {
          //   console.log("PUSH TO ARRAY");
          //   message[key]['y'].push(correct_prediction_ratio);
          //   message[key]['x'].push(message[key]['x'].length);
          // } else {
          //   console.log("ADD AN ARRAY");
          //   message[key]['y'] = Array([correct_prediction_ratio]);
          //   message[key]['x'] = Array([model["processed_samples"]]);
          // }

          // console.log("x="+message[key]['x']);
          // console.log("y="+message[key]['y']);
          var new_model_state = this.addPoint(model);
          console.log("New model state = " + new_model_state.x);
          new_state.push(new_model_state);
      }
    }   
    console.log("Message: " + message);
    this.setState({models: new_state});
    // this.setState({models: message})
    // console.log('After setState: ' + JSON.parse(this.state));
 

    this.ws_data_provider.onclose = () => {
      console.log('WS to data_provider: disconnected.')
    }

    this.ws_evaluation_server.onclose = () => {
      console.log('WS to evaluation_server: disconnected.')
    }

  }
}

  sendStartMessage = () => {
    console.log("Start")
    this.ws_data_provider.send('start');
    this.ws_evaluation_server.send('start');
  }

  // addPoint = () => {
    addPoint = (model) => {
    console.log("addPoint");
    var correct_prediction_ratio = model["correct_predictions"] / model["processed_samples"];
    if(model.hasOwnProperty('x')) {
      console.log("YES model.hasOwnProperty('x')");
      model['y'].push(correct_prediction_ratio);
      model['x'].push(model['processed_samples']);
    } else {
      console.log("NO model.hasOwnProperty('x')");
      model['y'] = Array([correct_prediction_ratio]);
      model['x'] = Array([model["processed_samples"]]);
    }
    return model;
  }


  render() {
    return (
      <div className="App">
          <div websocket={this.ws_data_provider} />
          <Header />
          <button onClick={this.sendStartMessage}>Start</button>
          <Models models={this.state.models} addPoint={this.addPoint}/>
      </div>
    );
  }
}

export default App;

