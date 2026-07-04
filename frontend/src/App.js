import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [papers, setPapers] = useState([]);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [analysis, setAnalysis] = useState(null);

  // Load papers on component mount
  useEffect(() => {
    loadPapers();
  }, []);

  const loadPapers = async () => {
    try {
      const response = await axios.get(`${API_URL}/papers`);
      setPapers(response.data);
    } catch (error) {
      console.error('Error loading papers:', error);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file first');
      return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/upload`, formData);
      alert(`✅ Paper uploaded successfully! ID: ${response.data.paper_id}`);
      loadPapers(); // Refresh the list
    } catch (error) {
      console.error('Upload error:', error);
      alert('❌ Upload failed: ' + (error.response?.data?.detail || error.message));
    }
    setLoading(false);
  };

  const handleSearch = async () => {
    if (!query) {
      alert('Please enter a search query');
      return;
    }
    
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/search`, {
        query: query,
        k: 5
      });
      setResults(response.data);
    } catch (error) {
      console.error('Search error:', error);
      alert('❌ Search failed');
    }
    setLoading(false);
  };

  const handleAsk = async () => {
    if (!query) {
      alert('Please enter a question');
      return;
    }
    if (!selectedPaper) {
      alert('Please select a paper first');
      return;
    }
    
    setLoading(true);
    setAnswer(''); // Clear previous answer
    try {
      const response = await axios.post(`${API_URL}/ask`, {
        question: query,
        paper_id: selectedPaper
      });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error('Question error:', error);
      alert('❌ Failed to get answer');
    }
    setLoading(false);
  };

  const handleAnalyze = async (paperId) => {
    if (!paperId) {
      alert('Please select a paper');
      return;
    }
    
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/analyze/${paperId}`);
      setAnalysis(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
      alert('❌ Analysis failed');
    }
    setLoading(false);
  };

  const styles = {
    container: { padding: '20px', fontFamily: 'Arial, sans-serif', maxWidth: '1200px', margin: '0 auto' },
    title: { color: '#2c3e50', borderBottom: '3px solid #3498db', paddingBottom: '10px' },
    section: { border: '1px solid #ddd', padding: '20px', margin: '20px 0', borderRadius: '8px', backgroundColor: '#f9f9f9' },
    sectionTitle: { color: '#34495e', marginTop: '0' },
    input: { width: '60%', padding: '10px', border: '1px solid #ddd', borderRadius: '4px', fontSize: '14px' },
    button: { padding: '10px 20px', marginLeft: '10px', backgroundColor: '#3498db', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '14px' },
    buttonDisabled: { padding: '10px 20px', marginLeft: '10px', backgroundColor: '#95a5a6', color: 'white', border: 'none', borderRadius: '4px', cursor: 'not-allowed', fontSize: '14px' },
    answerBox: { border: '2px solid #27ae60', padding: '20px', margin: '20px 0', borderRadius: '8px', backgroundColor: '#eafaf1' },
    resultItem: { borderBottom: '1px solid #eee', padding: '10px 0' },
    select: { padding: '10px', marginTop: '10px', border: '1px solid #ddd', borderRadius: '4px', width: '100%', maxWidth: '400px' },
    analysisBox: { border: '2px solid #8e44ad', padding: '20px', margin: '20px 0', borderRadius: '8px', backgroundColor: '#f4ecf7' },
    analysisItem: { margin: '10px 0', padding: '10px', borderLeft: '3px solid #8e44ad', paddingLeft: '15px' }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🤖 AI Research Assistant</h1>
      
      {/* Upload Section */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>📄 Upload PDF</h2>
        <input 
          type="file" 
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
          style={{ display: 'block', marginBottom: '10px' }}
        />
        <button 
          onClick={handleUpload} 
          disabled={loading}
          style={loading ? styles.buttonDisabled : styles.button}
        >
          {loading ? 'Uploading...' : 'Upload PDF'}
        </button>
        {papers.length > 0 && (
          <div style={{ marginTop: '10px' }}>
            <p>📚 {papers.length} papers uploaded</p>
          </div>
        )}
      </div>

      {/* Search & Ask Section */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>🔍 Search & Ask</h2>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your question or search query..."
          style={styles.input}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
        />
        <button 
          onClick={handleSearch} 
          disabled={loading}
          style={loading ? styles.buttonDisabled : { ...styles.button, backgroundColor: '#2ecc71' }}
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
        <button 
          onClick={handleAsk} 
          disabled={loading || !selectedPaper}
          style={loading || !selectedPaper ? styles.buttonDisabled : { ...styles.button, backgroundColor: '#e67e22' }}
        >
          {loading ? 'Thinking...' : 'Ask Question'}
        </button>
        
        <div>
          <select 
            onChange={(e) => setSelectedPaper(Number(e.target.value))} 
            value={selectedPaper || ''}
            style={styles.select}
          >
            <option value="">Select a paper</option>
            {papers.map(p => (
              <option key={p.id} value={p.id}>{p.title || `Paper ${p.id}`}</option>
            ))}
          </select>
          {selectedPaper && (
            <button 
              onClick={() => handleAnalyze(selectedPaper)} 
              disabled={loading}
              style={{ ...styles.button, backgroundColor: '#8e44ad', marginTop: '10px' }}
            >
              {loading ? 'Analyzing...' : 'Analyze Paper'}
            </button>
          )}
        </div>
      </div>

      {/* Answer Section */}
      {answer && (
        <div style={styles.answerBox}>
          <h3>💡 Answer:</h3>
          <p>{answer}</p>
        </div>
      )}

      {/* Analysis Section */}
      {analysis && (
        <div style={styles.analysisBox}>
          <h3>📊 Paper Analysis</h3>
          {analysis.summary && (
            <div style={styles.analysisItem}>
              <strong>Summary:</strong>
              <p>{analysis.summary}</p>
            </div>
          )}
          {analysis.key_contributions && (
            <div style={styles.analysisItem}>
              <strong>Key Contributions:</strong>
              <p>{analysis.key_contributions}</p>
            </div>
          )}
          {analysis.limitations && (
            <div style={styles.analysisItem}>
              <strong>Limitations:</strong>
              <p>{analysis.limitations}</p>
            </div>
          )}
          {analysis.future_work && (
            <div style={styles.analysisItem}>
              <strong>Future Work:</strong>
              <p>{analysis.future_work}</p>
            </div>
          )}
        </div>
      )}

      {/* Search Results */}
      {results.length > 0 && (
        <div style={styles.section}>
          <h3>📚 Search Results</h3>
          {results.map((result, i) => (
            <div key={i} style={styles.resultItem}>
              <p><strong>Relevance:</strong> {(result.similarity * 100).toFixed(2)}%</p>
              <p>{result.chunk.text.slice(0, 400)}...</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;