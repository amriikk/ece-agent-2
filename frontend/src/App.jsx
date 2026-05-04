import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

// ── Plotly: CSP-safe direct API call ─────────────────────────────────────────
function PlotlyChart({ figure }) {
  const divRef = useRef(null);

  useEffect(() => {
    if (!divRef.current || !figure?.data) return;
    if (!window.Plotly) return;

    window.Plotly.newPlot(
      divRef.current,
      figure.data,
      {
        ...figure.layout,
        autosize: true,
        height: 360,
        margin: { l: 52, r: 16, t: 48, b: 56 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: "'Special Elite', monospace", color: '#7eb8e8' },
        title: {
          ...figure.layout?.title,
          font: { family: "'Lora', serif", color: '#d4e6f5', size: 14 },
        },
        legend: {
          bgcolor: 'rgba(0,0,0,0)',
          font: { family: "'Special Elite', monospace", color: '#4a7fab', size: 11 },
          bordercolor: 'rgba(100,160,220,0.2)',
          borderwidth: 1,
        },
        xaxis: {
          ...figure.layout?.xaxis,
          gridcolor: 'rgba(100,160,220,0.1)',
          linecolor: 'rgba(100,160,220,0.25)',
          tickfont: { family: "'Special Elite', monospace", color: '#4a7fab', size: 10 },
          title: { ...figure.layout?.xaxis?.title, font: { color: '#7eb8e8', size: 11 } },
        },
        yaxis: {
          ...figure.layout?.yaxis,
          gridcolor: 'rgba(100,160,220,0.1)',
          linecolor: 'rgba(100,160,220,0.25)',
          tickfont: { family: "'Special Elite', monospace", color: '#4a7fab', size: 10 },
          title: { ...figure.layout?.yaxis?.title, font: { color: '#7eb8e8', size: 11 } },
        },
        colorway: ['#7eb8e8', '#6ea87e', '#e87e7e', '#c8a96e', '#9e7ec8', '#7ec8c8'],
      },
      {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['select2d', 'lasso2d'],
      }
    );

    return () => { if (divRef.current) window.Plotly.purge(divRef.current); };
  }, [figure]);

  return <div ref={divRef} style={{ width: '100%', height: '360px' }} />;
}

// ── Live status indicator (Improvement 4) ────────────────────────────────────
function StatusStream({ messages }) {
  if (!messages.length) return null;
  return (
    <div className="message-row ai">
      <div className="msg-stamp">PROCESSING</div>
      <div className="bubble ai status-bubble">
        {messages.map((m, i) => (
          <div key={i} className={`status-line ${i === messages.length - 1 ? 'active' : 'done'}`}>
            <span className="status-tick">{i === messages.length - 1 ? '▸' : '✓'}</span>
            {m}
          </div>
        ))}
        <span className="status-cursor">▌</span>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages]       = useState([]);
  const [input, setInput]             = useState('');
  const [datasetPath, setDatasetPath] = useState('datasets/data.csv');
  const [activeFile, setActiveFile]   = useState('data.csv');
  const [isLoading, setIsLoading]     = useState(false);
  const [statusLog, setStatusLog]     = useState([]);   // live node labels
  const [error, setError]             = useState(null);
  const [prevResult, setPrevResult]   = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);
  const readerRef      = useRef(null);   // holds active SSE reader for cleanup

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, statusLog, isLoading]);

  // ── File upload ─────────────────────────────────────────────────────────────
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setIsLoading(true);
    setError(null);

    try {
      const res  = await fetch('http://localhost:8000/api/upload', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('Upload failed');
      const data = await res.json();
      setDatasetPath(data.dataset_path);
      setActiveFile(data.filename);
      setPrevResult(null);
      setMessages(prev => [...prev, { role: 'system', content: `File loaded: **${data.filename}**` }]);
    } catch (err) {
      setError(err.message || 'Upload error.');
    } finally {
      setIsLoading(false);
      e.target.value = '';
    }
  };

  // ── SSE streaming submit (Improvement 4) ────────────────────────────────────
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput('');
    setError(null);
    setIsLoading(true);
    setStatusLog([]);
    setMessages(prev => [...prev, { role: 'user', content: userQuery }]);

    try {
      const res = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userQuery,
          dataset_path: datasetPath,
          chat_history: messages
            .filter(m => m.role !== 'system')
            .map(m => ({ role: m.role, content: m.content })),
          prev_result: prevResult,
        }),
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);

      const reader = res.body.getReader();
      readerRef.current = reader;
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop(); // keep incomplete chunk

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          let event;
          try { event = JSON.parse(raw); }
          catch { continue; }

          if (event.type === 'progress') {
            setStatusLog(prev => [...prev, event.message]);
          } else if (event.type === 'done') {
            setStatusLog([]);
            setMessages(prev => [...prev, {
              role: 'ai',
              content: event.final_answer,
              visualization: event.visualization_figure,
            }]);
            setPrevResult(event.execution_result);
          } else if (event.type === 'error') {
            throw new Error(event.message);
          }
        }
      }
    } catch (err) {
      setStatusLog([]);
      setError(err.message || 'Request failed.');
    } finally {
      setIsLoading(false);
      readerRef.current = null;
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(e); }
  };

  return (
    <div className="root">
      <div className="scanlines" aria-hidden="true" />

      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <span className="logo-bracket">[</span>
          <h1 className="logo-text">DATUM</h1>
          <span className="logo-bracket">]</span>
          <span className="logo-sub">data analysis terminal</span>
        </div>
        <div className="header-right">
          <div className="file-indicator">
            <span className="file-label">ACTIVE FILE</span>
            <span className="file-name">{activeFile}</span>
          </div>
          <div className="path-row">
            <span className="path-prefix">PATH:</span>
            <input
              className="path-input"
              type="text"
              value={datasetPath}
              onChange={e => setDatasetPath(e.target.value)}
              spellCheck={false}
            />
          </div>
          <input type="file" id="csv-upload" accept=".csv" style={{ display: 'none' }} onChange={handleFileUpload} />
          <label htmlFor="csv-upload" className="upload-btn">⊕ LOAD CSV</label>
        </div>
      </header>

      {/* ── Chat pane ── */}
      <main className="chat-pane">
        {messages.length === 0 && !isLoading && (
          <div className="empty-state">
            <div className="empty-glyph">◈</div>
            <p className="empty-title">Awaiting query.</p>
            <p className="empty-hint">Load a CSV and ask anything — overviews, filters, charts, comparisons.</p>
            <div className="example-queries">
              {[
                'What does this file contain?',
                'Show me the top 5 rows',
                'Plot revenue by region as a bar chart',
                'Filter rows where status is active',
              ].map(q => (
                <button key={q} className="example-chip" onClick={() => { setInput(q); inputRef.current?.focus(); }}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => {
          if (msg.role === 'system') return (
            <div key={i} className="system-notice">
              <span className="sys-icon">▸</span>
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          );
          if (msg.role === 'user') return (
            <div key={i} className="message-row user">
              <div className="bubble user">
                <span className="prompt-caret">&gt;&gt;</span>{msg.content}
              </div>
            </div>
          );
          return (
            <div key={i} className="message-row ai">
              <div className="msg-stamp">FILED</div>
              <div className="bubble ai">
                <div className="ai-content"><ReactMarkdown>{msg.content}</ReactMarkdown></div>
                {msg.visualization?.data && (
                  <div className="chart-container">
                    <div className="chart-header"><span className="chart-tag">◈ VISUALISATION</span></div>
                    <PlotlyChart figure={msg.visualization} />
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Live streaming status (replaces plain spinner) */}
        {isLoading && <StatusStream messages={statusLog} />}

        {error && <div className="error-strip"><span className="error-icon">✕</span> {error}</div>}
        <div ref={messagesEndRef} />
      </main>

      {/* ── Input footer ── */}
      <footer className="input-footer">
        <div className="input-row">
          <span className="input-caret">▸</span>
          <textarea
            ref={inputRef}
            className="query-textarea"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter query or ask a general question about your data..."
            disabled={isLoading}
            rows={1}
          />
          <button className="send-btn" onClick={handleSubmit} disabled={isLoading || !input.trim()}>
            {isLoading ? '◌' : 'RUN ▸'}
          </button>
        </div>
        <div className="footer-meta">
          <span>SHIFT+ENTER for new line · ENTER to submit</span>
          <span className={`status-dot ${isLoading ? 'busy' : 'ready'}`}>
            {isLoading ? '● PROCESSING' : '● READY'}
          </span>
        </div>
      </footer>
    </div>
  );
}
