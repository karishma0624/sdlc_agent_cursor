import { useState, useEffect } from 'react';
import { Activity, Server, Database, Code, Zap } from 'lucide-react';
import BuildForm from './components/BuildForm';
import BuildStatus from './components/BuildStatus';

const API_BASE = '/api'; // Proxy handles this in dev, or relative in prod

function App() {
  const [activeJob, setActiveJob] = useState(null);
  const [health, setHealth] = useState('checking');
  const [providers, setProviders] = useState({});

  useEffect(() => {
    // Check backend health
    fetch(`${API_BASE}/health`)
      .then(r => setHealth(r.ok ? 'online' : 'offline'))
      .catch(() => setHealth('offline'));

    // Check AI providers
    fetch(`${API_BASE}/providers`)
      .then(r => r.json())
      .then(d => setProviders(d.providers || {}))
      .catch(() => { });
  }, []);

  const handleBuildStarted = (jobId) => {
    setActiveJob(jobId);
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-blue-600 p-2 rounded-lg text-white">
              <Code size={20} />
            </div>
            <h1 className="text-xl font-bold tracking-tight">SDLC Builder Agent</h1>
          </div>

          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 rounded-full">
              <Server size={14} className="text-gray-500" />
              <span className={`w-2 h-2 rounded-full ${health === 'online' ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <span className="font-medium text-gray-600">{health.toUpperCase()}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        <div className="grid gap-8 lg:grid-cols-12">
          {/* Left Column: Input */}
          <div className="lg:col-span-7 space-y-8">
            <section>
              <div className="mb-4">
                <h2 className="text-2xl font-bold text-gray-800">What shall we build?</h2>
                <p className="text-gray-500">Describe your application requirements in detail.</p>
              </div>
              <BuildForm onBuildStarted={handleBuildStarted} />
            </section>

            {/* AI Capability Status */}
            <section>
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">AI Providers</h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {Object.entries(providers).map(([k, v]) => (
                  <div key={k} className={`flex items-center gap-2 p-3 rounded-lg border text-sm font-medium
                                ${v ? 'bg-white border-gray-200 text-gray-700' : 'bg-gray-50 border-gray-100 text-gray-400'}`}>
                    <Zap size={14} className={v ? 'text-yellow-500' : 'text-gray-300'} />
                    {k}
                  </div>
                ))}
                {Object.keys(providers).length === 0 && (
                  <div className="col-span-full text-sm text-gray-400 italic">No provider info available</div>
                )}
              </div>
            </section>
          </div>

          {/* Right Column: Status & History */}
          <div className="lg:col-span-5">
            {activeJob ? (
              <BuildStatus jobId={activeJob} />
            ) : (
              <div className="h-full flex flex-col items-center justify-center p-12 text-center text-gray-400 border-2 border-dashed border-gray-200 rounded-xl">
                <Activity size={48} className="mb-4 opacity-50" />
                <h3 className="text-lg font-medium text-gray-600">Ready to build</h3>
                <p className="text-sm">Enter a prompt to start the autonomous agent.</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
