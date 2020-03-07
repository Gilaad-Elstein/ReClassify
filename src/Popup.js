import React from 'react';  
import './App.css';

class Popup extends React.Component {  
  render() {  
return (  
<div className='popup'>  
<div className='popup\_inner'>  
<h1>{this.props.header}</h1>

{this.props.text.split('~n').map( (it, i) => <div key={'x'+i}>{it}<br></br></div>)}

<br></br><br></br>

<button onClick={this.props.closePopup}>Got it</button>  
<br></br><br></br>
</div>  
</div>  
);  
}  
}  

export default Popup;