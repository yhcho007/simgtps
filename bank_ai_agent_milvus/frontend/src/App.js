import React from 'react';
import Login from './Login';
import Chat from './Chat';
export default function App(){ const [token,setToken]=React.useState(null); return (<div style={{padding:20}}>{token? <Chat token={token}/>: <Login onToken={t=>setToken(t)}/>}</div>); }
