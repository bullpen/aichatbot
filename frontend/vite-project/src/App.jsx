import React from 'react';  // 추가!
import { useState, useRef, useEffect } from "react";

export default function App() {
  const [message, setMessage] = useState("");
  const [chatLog, setChatLog] = useState([{ role: "bot", text: "안녕하세요.\r\n 영화 리뷰 관련 긍정/부정 판단 봇입니다. 영화리뷰 평을 작성하시면 긍정/부정을 판단하여 알려드립니다. 아래의 메시지를 사용하실 수 있습니다.\r\n\r\n 1. 이 영화는 재미 없었어\r\n2. 최신 영화 목록 보여줘\r\n3. 승부 영화에 대해 소개해줘" }]);
  const [loading, setLoading] = useState(false);
  const chatContainerRef = useRef(null);

  const encodeHTML = (text) => {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;')
      .replace(/\n/g, '<br />');
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatLog, loading]);

  const sendMessage = async () => {
    if (!message.trim()) return;

    const newChatLog = [...chatLog, { role: "user", text: message }];
    setChatLog(newChatLog);
    setMessage("");
    setLoading(true);

    try {
      const res = await fetch("http://ownarea.synology.me:8001/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
      });

      const data = await res.json();
      setChatLog([...newChatLog, { role: "bot", text: data.reply, data: data }]);
    } catch (err) {
      setChatLog([...newChatLog, { role: "bot", text: "[서버 응답 실패]" }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-700 flex flex-col items-center p-4">
      <div className="w-full max-w-xl bg-gray-300 shadow-lg rounded-lg p-4 flex flex-col gap-4 h-[90vh]">
        <h1 className="text-2xl font-bold text-center text-blue-600">🎬 감정 챗봇</h1>

        <div 
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto space-y-2 border p-2 rounded bg-gray-500"
        >
          {chatLog.map((msg, idx) => (
            <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div className={`text-sm p-2 rounded max-w-[80%] ${msg.role === 'user' ? 'bg-blue-100' : 'bg-gray-200'}`}>
                <div dangerouslySetInnerHTML={{ __html: encodeHTML(msg.text) }} />
              </div>
              {msg.data?.mode && (
                <div className="text-xs text-gray-100 mt-1">
                  {msg.data.mode}
                </div>
              )}
            </div>
          ))}
          {loading && <div className="text-sm text-gray-300 italic">...응답 생성 중</div>}
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            className="flex-1 border border-gray-300 rounded px-3 py-2 focus:outline-none"
            placeholder="메시지를 입력하세요"
          />
          <button
            onClick={sendMessage}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            전송
          </button>
        </div>
      </div>
    </div>
  );
}
