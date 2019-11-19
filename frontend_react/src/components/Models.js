import React, { Component } from 'react';
import ModelEval from './ModelEval';

class Models extends Component {

  render() {
    if(this.props.models.length > 0) {
      return this.props.models.map((model) => (
        <ModelEval key={model.id} model={model} addPoint={this.props.addPoint}/>
    ))
    } else {
      return (<div>No models</div>)
    }
    
  }
}

export default Models;
