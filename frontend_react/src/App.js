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
    // on connecting, do nothing but log it to the console
    console.log('WS to data_provider: connected.')
    }

    this.ws_evaluation_server.onopen = () => {
      // on connecting, do nothing but log it to the console
    console.log('WS to evaluation_server: connected.')
    }

    this.ws_evaluation_server.onmessage = evt => {
    // listen to data sent from the websocket server
    const message = JSON.parse(evt.data)
    console.log(evt.data);
    this.setState({models: message})
    console.log('After setState: ' + this.state);
    // for(var key in message) {
    //   if(message.hasOwnProperty(key)) {
    //       var model = message[key];
    //       console.log(model);
    //       var correct_prediction_ratio = model["correct_predictions"] / model["processed_samples"];
    //       console.log('correct_prediction_ratio = ' + correct_prediction_ratio);
    //       var id = model['id'];
    //       this.setState();
    //   }
    // }    

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


  render() {
    return (
      <div className="App">
          <div websocket={this.ws_data_provider} />
          <Header />
          <button onClick={this.sendStartMessage}>Start</button>
          <Models models={this.state.models} />
      </div>
    );
  }
}

export default App;
