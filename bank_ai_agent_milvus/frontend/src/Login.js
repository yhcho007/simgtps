import React, {useState} from 'react';
import axios from 'axios';

export default function Login({onToken}){
  const [u,setU]=useState('alice'); const [p,setP]=useState('password');
  async function submit(){
    try{
      const r = await axios.post('http://localhost:8000/login', {username:u,password:p});
      onToken(r.data.access_token);
    }catch(e){ alert('Login failed'); }
  }
  return (
    <div style={{maxWidth:400}}>
      <h3>Login (demo)</h3>
      <div><input value={u} onChange={e=>setU(e.target.value)} /></div>
      <div><input value={p} onChange={e=>setP(e.target.value)} type='password' /></div>
      <button onClick={submit}>로그인</button>
    </div>
  );
}
