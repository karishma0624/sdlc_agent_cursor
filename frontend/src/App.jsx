import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';

function App() {
  const [selectedRunId, setSelectedRunId] = useState(null);

  const handleNewChat = () => {
    setSelectedRunId(null);
  };

  const handleSelectRun = (id) => {
    setSelectedRunId(id);
  };

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      <Sidebar
        onNewChat={handleNewChat}
        onSelectRun={handleSelectRun}
        selectedRunId={selectedRunId}
      />

      <div className="flex-1 h-full min-w-0">
        <ChatInterface
          key={selectedRunId || 'new'} // Force remount on change
          runId={selectedRunId}
          setRunId={setSelectedRunId}
        />
      </div>
    </div>
  );
}

export default App;
