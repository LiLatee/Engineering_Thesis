import React, { Component } from 'react';
import ModelEval from './ModelEval';

class Models extends Component {

  render() {
    if(this.props.models.length > 0) {
      console.log("this.props.models.length = " + this.props.models.length);
      return this.props.models.map((model) => (
        <ModelEval key={model.id} model={model}/>
    ))
    } else {
      console.log("this.props.models.length = " + this.props.models.length);
      return (<div>No models</div>)
    }
    
  }
}

export default Models;
