import React, { useState, useRef, useEffect } from 'react';
import Plot from 'react-plotly.js';
import './App.css'; 

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [datasetPath, setDatasetPath] = useState('data.csv'); // Default path
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // This is the crucial piece for memory: holding the structured state artifact
  const [prevResult, setPrevResult] = useState(null); 
  
  const messagesEndRef = useRef(null);

  // Auto-scroll to the bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  useEffect(() => { scrollToBottom(); }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userQuery = input;
    setInput('');
    setError(null);
    setIsLoading(true);

    // Optimistically add user message to UI
    const updatedMessages = [...messages, { role: 'user', content: userQuery }];
    setMessages(updatedMessages);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userQuery,
          dataset_path: datasetPath,
          // Send previous chat history (excluding visualizations for payload size)
          chat_history: messages.map(m => ({ role: m.role, content: m.content })),
          prev_result: prevResult 
        }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();

      // Update state with AI response, visualization, and the new execution result artifact
      setMessages(prev => [
        ...prev, 
        { 
          role: 'ai', 
          content: data.final_answer, 
          visualization: data.visualization_figure 
        }
      ]);
      setPrevResult(data.execution_result); 

    } catch (err) {
      setError(err.message || 'An error occurred during execution.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Data Analyst Core</h1>
        <div className="config-bar">
          <label>Dataset Path:</label>
          <input 
            type="text" 
            value={datasetPath} 
            onChange={(e) => setDatasetPath(e.target.value)}
            className="tech-input"
          />
        </div>
      </header>

      <main className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message-wrapper ${msg.role}`}>
            <div className={`message-bubble ${msg.role}`}>
              <div className="message-content">{msg.content}</div>
              
              {/* Conditionally render the Plotly chart if it exists */}
              {msg.visualization && (
                <div className="visualization-container">
                  <Plot
                    data={msg.visualization.data}
                    layout={{ 
                      ...msg.visualization.layout, 
                      autosize: true,
                      margin: { l: 40, r: 20, t: 40, b: 40 },
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      font: { family: 'Inter, sans-serif', color: '#e2e8f0' } // Matching technical UI
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '100%' }}
                  />
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message-wrapper ai">
            <div className="message-bubble ai loading-indicator">
              <span className="dot"></span><span className="dot"></span><span className="dot"></span>
            </div>
          </div>
        )}
        {error && <div className="error-banner">Error: {error}</div>}
        <div ref={messagesEndRef} />
      </main>

      <footer className="input-area">
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Query the dataset (e.g., 'Show the top 5 rows')"
            disabled={isLoading}
            className="tech-input query-input"
          />
          <button type="submit" disabled={isLoading || !input.trim()} className="tech-button">
            EXECUTE
          </button>
        </form>
      </footer>
    </div>
  );
}