import React, { Component } from 'react';
import Plot from 'react-plotly.js';

class ModelEval extends Component {

  // constructor(props) {
  //   super(props);

  //   this.state = this.getInitialState();
  //   console.log(this.state);
  // }

  // // static getDerivedStateFromProps(props, state) {
  // // componentDidMount() {
  // componentDidUpdate(nextProps) {
  //   console.log("Current state: "+this.state["processed_samples"]);
  //   console.log("Next props: "+nextProps.model["processed_samples"]);
  //   if(this.state["processed_samples"] != nextProps.model["processed_samples"]) {
  //     console.log("State is changing.");
  //     this.setState(this.getUpdatedState);
  //   } else {
  //     console.log("State is the same.");
  //   }
  //   // console.log("nextProps = " + nextProps.value);
  //   // this.props.addPoint.bind(this);
  //   // return this.getInitialState();
  //   // this.setState(this.getUpdatedState);
  // }

  // getInitialState = () => {
  //   let model = this.props.model;
  //   // if(model.hasOwnProperty('x')) {
  //   //   console.log("PUSH TO ARRAY");
  //   //   model['y'].push(correct_prediction_ratio);
  //   //   model['x'].push(model['processed_samples']);
  //   // } else {
  //   //   console.log("ADD AN ARRAY");
  //   //   model['y'] = Array([correct_prediction_ratio]);
  //   //   model['x'] = Array([model["processed_samples"]]);
  //   // }
  //   model['y'] = Array([model["correct_predictions"] / model["processed_samples"]]);
  //   model['x'] = Array([model["processed_samples"]]);
  //   return model;
  // };

  // getUpdatedState = () => {
  //   let model = this.props.model;
  //   var correct_prediction_ratio = model["correct_predictions"] / model["processed_samples"];
  //   model['y'].push(correct_prediction_ratio);
  //   model['x'].push(model['processed_samples']);
  //   return model;
  // };

  render() {
      const {id, x, y, processed_samples, correct_predictions, roc_auc_score} = this.props.model;
      console.log("ModelEval processed_samples = " + processed_samples);
      console.log("ModelEval x="+x);
      console.log("ModelEval y="+y);
      // this.props.addPoint.bind(this, x, y, processed_samples, correct_predictions);
      return (
        <Plot
            data={[
            {
                x: x,
                y: y,
                type: 'scatter',
                mode: 'lines+points',
                marker: {color: 'blue'},
            },
            ]}
            layout={ {width: 1500, height: 400, title: "Model " + id} }
        />
        );
  }
}

export default ModelEval
