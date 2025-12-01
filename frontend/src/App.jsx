import React, { useState } from 'react';
import UploadSection from './components/UploadSection';
import ResultsTable from './components/ResultsTable';
import { FileText, Camera, Upload } from 'lucide-react';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-gray-50 p-8 font-sans text-gray-900">
      <header className="mb-10 text-center">
        <h1 className="text-4xl font-extrabold tracking-tight text-blue-600 mb-2">
          OMR Grader <span className="text-gray-400 font-light">Pro</span>
        </h1>
        <p className="text-gray-500">Bulk Process & Grade OMR Sheets Instantly</p>
      </header>

      <main className="max-w-5xl mx-auto space-y-8">
        <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
          <UploadSection setResults={setResults} setLoading={setLoading} />
        </div>

        {loading && (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-500 animate-pulse">Processing OMR Sheets...</p>
          </div>
        )}

        {results && (
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <ResultsTable results={results} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
