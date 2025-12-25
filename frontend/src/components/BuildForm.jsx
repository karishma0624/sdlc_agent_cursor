import { useState } from 'react';

export default function BuildForm({ onBuildStarted }) {
    const [prompt, setPrompt] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!prompt.trim()) return;

        setLoading(true);
        setError(null);

        try {
            const res = await fetch('/api/sdlc/build', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt }),
            });

            if (!res.ok) throw new Error('Failed to start build');

            const data = await res.json();
            if (onBuildStarted) {
                onBuildStarted(data.job_id);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Create New Project</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 mb-1">
                        Build Prompt
                    </label>
                    <textarea
                        id="prompt"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Describe the application you want to build (e.g., 'A todo list app with React and FastAPI')..."
                        className="w-full h-32 px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all"
                        disabled={loading}
                    />
                </div>

                {error && (
                    <div className="p-3 bg-red-50 text-red-700 text-sm rounded-lg">
                        {error}
                    </div>
                )}

                <button
                    type="submit"
                    disabled={loading || !prompt.trim()}
                    className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-colors
            ${loading || !prompt.trim()
                            ? 'bg-blue-400 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700 shadow-md hover:shadow-lg'
                        }`}
                >
                    {loading ? 'Starting Build...' : 'Start Build'}
                </button>
            </form>
        </div>
    );
}
