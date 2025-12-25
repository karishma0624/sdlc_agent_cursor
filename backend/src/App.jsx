import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [message, setMessage] = useState('Loading...')

  useEffect(() => {
    fetch('/api/health')
      .then(response => response.json())
      .then(data => setMessage(Backend status: ))
      .catch(() => setMessage('Failed to connect to backend'))
  }, [])

  return (
    <div className="app">
      <h1>SDLC Builder Agent</h1>
      <p>{message}</p>
    </div>
  )
}

export default App
