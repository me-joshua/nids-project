import React, { useState, useRef, useEffect } from 'react';
import { Send, Shield, AlertTriangle, CheckCircle, XCircle, Loader2, Bot, User, Trash2 } from 'lucide-react';
import axios from 'axios';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://nids-project.onrender.com';

const App = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'system',
      content: 'Welcome to NEXUS AI Network Intrusion Detection System. Paste a network packet data line and I\'ll analyze it for potential threats using our advanced multimodal ensemble AI models.',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking', 'online', 'offline'
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Check backend health on component mount
  const checkBackendHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/health`, {
        timeout: 10000 // 10 second timeout
      });
      setBackendStatus('online');
      console.log('Backend health check successful:', response.data);
    } catch (error) {
      setBackendStatus('offline');
      console.error('Backend health check failed:', error);
      
      // Add a system message about backend status
      const statusMessage = {
        id: Date.now(),
        type: 'system',
        content: `Backend service connection failed. You can still use the interface, but packet analysis may not work until the backend is available. (${API_BASE_URL})`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, statusMessage]);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    checkBackendHealth();
  }, []);

  const getThreatLevel = (confidence) => {
    if (confidence >= 0.9) return { level: 'HIGH', color: 'threat-high', glow: 'threat-glow-high' };
    if (confidence >= 0.7) return { level: 'MEDIUM', color: 'threat-medium', glow: 'threat-glow-medium' };
    if (confidence >= 0.5) return { level: 'LOW', color: 'threat-low', glow: 'threat-glow-low' };
    return { level: 'NORMAL', color: 'threat-normal', glow: 'threat-glow-normal' };
  };

  const formatAnalysisResult = (analysis) => {
    const threat = getThreatLevel(analysis.ensemble_confidence);
    
    return (
      <div className="space-y-4">
        <div className={`p-4 rounded-lg border-2 ${threat.glow}`} style={{ borderColor: `var(--tw-color-${threat.color})` }}>
          <div className="flex items-center gap-2 mb-2">
            {analysis.is_attack ? (
              <XCircle className={`h-5 w-5 text-${threat.color}`} />
            ) : (
              <CheckCircle className="h-5 w-5 text-threat-normal" />
            )}
            <span className={`font-bold text-${threat.color}`}>
              {analysis.is_attack ? `THREAT DETECTED - ${threat.level}` : 'NORMAL TRAFFIC'}
            </span>
          </div>
          <div className="text-sm text-chat-text">
            <p><strong>Ensemble Confidence:</strong> {(analysis.ensemble_confidence * 100).toFixed(2)}%</p>
            <p><strong>Attack Probability:</strong> {(analysis.attack_probability * 100).toFixed(2)}%</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-chat-input p-4 rounded-lg">
            <h4 className="font-semibold mb-2 text-blue-400">Base Models</h4>
            <div className="space-y-1 text-sm">
              {Object.entries(analysis.base_models).map(([model, pred]) => (
                <div key={model} className="flex justify-between">
                  <span className="capitalize">{model.replace('_', ' ')}:</span>
                  <span className={pred.prediction === 1 ? 'text-red-400' : 'text-green-400'}>
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-chat-input p-4 rounded-lg">
            <h4 className="font-semibold mb-2 text-purple-400">Ensemble Models</h4>
            <div className="space-y-1 text-sm">
              {Object.entries(analysis.ensemble_models).map(([model, pred]) => (
                <div key={model} className="flex justify-between">
                  <span className="capitalize">{model.replace('_', ' ')}:</span>
                  <span className={pred.prediction === 1 ? 'text-red-400' : 'text-green-400'}>
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-chat-input p-4 rounded-lg">
          <h4 className="font-semibold mb-2 text-cyan-400">Deep Learning Analysis</h4>
          <div className="text-sm">
            {analysis.deep_learning.confidence !== null && analysis.deep_learning.confidence !== undefined ? (
              <>
                <p><strong>Multimodal Confidence:</strong> {(analysis.deep_learning.confidence * 100).toFixed(2)}%</p>
                <p><strong>Modalities Processed:</strong> {analysis.deep_learning.modalities_count}</p>
              </>
            ) : (
              <p className="text-gray-400">Deep learning model not available or failed to process.</p>
            )}
          </div>
        </div>

        {analysis.processing_time && (
          <div className="text-xs text-gray-400 text-center">
            Analysis completed in {analysis.processing_time.toFixed(3)}s using {analysis.models_used} models
          </div>
        )}
      </div>
    );
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/analyze`, {
        packet_data: inputValue.trim()
      });

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.data,
        timestamp: new Date(),
        isAnalysis: true
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      let errorMessage = 'Failed to analyze packet. Please try again.';
      let errorDetails = error.message;

      if (error.code === 'NETWORK_ERROR' || error.message.includes('Network Error')) {
        errorMessage = 'Cannot connect to the NIDS backend service. Please check your internet connection and try again.';
        errorDetails = `Backend URL: ${API_BASE_URL}`;
      } else if (error.response?.status === 404) {
        errorMessage = 'API endpoint not found. The backend service may be starting up.';
      } else if (error.response?.status >= 500) {
        errorMessage = 'Backend server error. The service may be temporarily unavailable.';
      } else if (error.response?.data?.error) {
        errorMessage = error.response.data.error;
      }

      const errorMessageObj = {
        id: Date.now() + 1,
        type: 'bot',
        content: {
          error: true,
          message: errorMessage,
          details: errorDetails
        },
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessageObj]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: 1,
        type: 'system',
        content: 'Welcome to NAS Technology Network Intrusion Detection System. Paste a network packet data line and I\'ll analyze it for potential threats using our advanced multimodal ensemble AI models.',
        timestamp: new Date()
      }
    ]);
  };

  const samplePackets = [
    "0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.11,0.00,0.00,0.00,0.00,0.00",
    "0,tcp,smtp,SF,1,46,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,1,1,1.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00",
    "0,tcp,ftp_data,SF,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,2,2,1.00,0.00,0.50,0.00,0.00,0.00,0.00,0.00"
  ];

  return (
    <div className="flex h-screen bg-chat-bg text-chat-text">
      {/* Sidebar */}
      <div className="w-64 bg-chat-sidebar border-r border-chat-border p-4 flex flex-col">
        <div className="flex items-center gap-2 mb-6">
          <Shield className="h-8 w-8 text-blue-400" />
          <div className="flex-1">
            <h1 className="font-bold text-lg">NEXUS AI</h1>
            <p className="text-xs text-gray-400">NIDS AI System</p>
          </div>
          <div className="flex flex-col items-center">
            <div className={`w-3 h-3 rounded-full ${
              backendStatus === 'online' ? 'bg-green-400' : 
              backendStatus === 'offline' ? 'bg-red-400' : 
              'bg-yellow-400'
            }`}></div>
            <span className="text-xs text-gray-400 mt-1">
              {backendStatus === 'online' ? 'Online' : 
               backendStatus === 'offline' ? 'Offline' : 
               'Checking'}
            </span>
          </div>
        </div>

        <button
          onClick={clearChat}
          className="flex items-center gap-2 w-full p-3 mb-4 bg-chat-input rounded-lg hover:bg-gray-600 transition-colors"
        >
          <Trash2 className="h-4 w-4" />
          Clear Chat
        </button>

        <div className="flex-1">
          <h3 className="font-semibold mb-3 text-sm">Sample Packets</h3>
          <div className="space-y-2">
            {samplePackets.map((packet, index) => (
              <button
                key={index}
                onClick={() => setInputValue(packet)}
                className="w-full p-2 text-left text-xs bg-chat-input rounded hover:bg-gray-600 transition-colors break-all"
              >
                {packet.substring(0, 50)}...
              </button>
            ))}
          </div>
        </div>

        <div className="mt-4 p-3 bg-chat-input rounded-lg">
          <div className="text-xs text-gray-400">
            <p className="font-semibold mb-1">AI Models Active:</p>
            <p>• Multimodal Deep Learning</p>
            <p>• 6 Base ML Models</p>
            <p>• 2 Ensemble Models</p>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b border-chat-border p-4 bg-chat-sidebar">
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-blue-400" />
            <h2 className="font-semibold">Network Intrusion Detection Analysis</h2>
            <div className="ml-auto flex items-center gap-2 text-sm text-gray-400">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              AI Models Online
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-hide">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.type !== 'user' && (
                <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
                  {message.type === 'system' ? (
                    <Shield className="h-4 w-4" />
                  ) : (
                    <Bot className="h-4 w-4" />
                  )}
                </div>
              )}
              
              <div className={`max-w-3xl ${message.type === 'user' ? 'bg-blue-600' : 'bg-chat-input'} rounded-lg p-4`}>
                {message.type === 'user' ? (
                  <div className="font-mono text-sm break-all">{message.content}</div>
                ) : message.isAnalysis ? (
                  message.content.error ? (
                    <div className="text-red-400">
                      <p className="font-semibold">Error</p>
                      <p className="text-sm">{message.content.message}</p>
                      {message.content.details && (
                        <p className="text-xs text-gray-400 mt-1">{message.content.details}</p>
                      )}
                    </div>
                  ) : (
                    formatAnalysisResult(message.content)
                  )
                ) : (
                  <div className="prose prose-invert max-w-none">
                    {message.content}
                  </div>
                )}
                
                <div className="text-xs text-gray-400 mt-2">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>

              {message.type === 'user' && (
                <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center flex-shrink-0">
                  <User className="h-4 w-4" />
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
                <Bot className="h-4 w-4" />
              </div>
              <div className="bg-chat-input rounded-lg p-4">
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Analyzing packet with multimodal AI models...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-chat-border p-4">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Paste network packet data here for analysis..."
              className="flex-1 bg-chat-input border border-chat-border rounded-lg px-4 py-3 focus:outline-none focus:border-blue-400 transition-colors"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !inputValue.trim()}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-6 py-3 rounded-lg transition-colors flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
              Analyze
            </button>
          </form>
          
          <div className="mt-2 text-xs text-gray-400">
            Paste a comma-separated network packet data line from NSL-KDD dataset format
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
