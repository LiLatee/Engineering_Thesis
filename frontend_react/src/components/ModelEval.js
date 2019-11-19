import React, { Component } from 'react';
import Plot from 'react-plotly.js';

class ModelEval extends Component {
  render() {
      const {id, x, y, processed_samples} = this.props.model;
      console.log("ModelEval processed_samples = " + processed_samples);
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
