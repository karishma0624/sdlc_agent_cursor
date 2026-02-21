import { useState } from 'react'
import { Layout } from 'lucide-react'

export default function App() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center p-8 bg-white rounded-xl shadow-lg border max-w-md mx-auto">
        <div className="bg-blue-50 p-4 rounded-full inline-flex mb-4">
            <Layout className="w-8 h-8 text-blue-600" />
        </div>
        <h1 className="text-2xl font-bold mb-2">Frontend Generated (Fallback)</h1>
        <p className="text-gray-600 mb-6">
          The AI providers (Gemini/v0) failed to generate the specific design, so we created this working React + Vite + Tailwind scaffold for you.
        </p>
        <p className="text-xs text-gray-500 font-mono bg-gray-100 p-2 rounded">
          Edit src/App.jsx to start building.
        </p>
      </div>
    </div>
  )
}
