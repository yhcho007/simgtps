import React, {useState, useEffect, useRef} from 'react';
import axios from 'axios';

export default function Chat({token}){
  const [msgs, setMsgs] = useState([]);
  const [text, setText] = useState('');
  const wsRef = useRef(null);

  useEffect(()=>{
    const ws = new WebSocket('ws://localhost:8000/ws/chat');
    ws.onopen = ()=>console.log('ws open');
    ws.onmessage = (e)=>{
      const data = JSON.parse(e.data);
      // expect {reply, msg_id}
      setMsgs(m => [...m, {from:'agent', text: data.reply, msg_id: data.get('msg_id') || data.msg_id}]);
    };
    wsRef.current = ws;
    return ()=>ws.close();
  },[]);

  async function send(){
    if(!text) return;
    const payload = {session_id:'local-demo', text, user:{id:'demo_user', account_id:'111-222-333'}};
    setMsgs(m=>[...m, {from:'user', text}]);
    try{
      wsRef.current.send(JSON.stringify(payload));
    }catch(e){
      const r = await axios.post('http://localhost:8000/chat/send', payload, {headers:{Authorization:`Bearer ${token}`}});
      setMsgs(m=>[...m, {from:'agent', text: r.data.reply, msg_id: r.data.msg_id}]);
    }
    setText('');
  }

  // submit feedback (rating 1-10). If rating<=3 and no reason, backend will reject
  async function submitFeedback(msg_id, rating, reason){
    try{
      const payload = {session_id:'local-demo', msg_id, rating, reason};
      await axios.post('http://localhost:8000/chat/feedback', payload, {headers:{Authorization:`Bearer ${token}`}});
      alert('Feedback recorded. 감사합니다.');
    }catch(e){
      alert('Feedback error: ' + (e.response?.data?.detail || e.message));
    }
  }

  return (
    <div style={{maxWidth:700}}>
      <div style={{border:'1px solid #ddd', padding:10, minHeight:360, background:'#fff'}}>
        {msgs.map((m,i)=> (
          <div key={i} style={{display:'flex', justifyContent: m.from==='user' ? 'flex-end' : 'flex-start', margin:'10px 0'}}>
            <div style={{maxWidth: '70%', padding:12, borderRadius:12, background: m.from==='user' ? '#cfe9ff' : '#f1f1f1'}}>
              <div style={{fontSize:14}}>{m.text}</div>
              {m.from==='agent' && (
                <div style={{marginTop:8}}>
                  <label>만족도 (1-10): </label>
                  {[1,2,3,4,5,6,7,8,9,10].map(v=> (
                    <button key={v} onClick={()=>{
                      const reason = v<=3 ? prompt('불만 이유를 간단히 적어주세요 (필수)') || '' : '';
                      submitFeedback(m.msg_id, v, reason);
                    }} style={{marginLeft:4}}>{v}</button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      <div style={{marginTop:8, display:'flex'}}>
        <input value={text} onChange={e=>setText(e.target.value)} style={{flex:1, padding:10}} placeholder="메시지 입력..." />
        <button onClick={send} style={{marginLeft:8, padding:'10px 14px'}}>전송</button>
      </div>
    </div>
  );
}
