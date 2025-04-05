import React, { useState } from 'react';
import Header from './components/Header';
import EmojiForm from './components/EmojiForm';
import EmojiOutput from './components/EmojiOutput';
import axios from 'axios';
import './App.css';

function App() {
  const [genImage, setGenImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async (prompt, imgType) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/generate', {
        "prompt": prompt,
        "model_type": imgType
      });

      setGenImage(response.data.image);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate emoji');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Header />
      <EmojiForm onGenerate={handleGenerate} isLoading={isLoading} />
      <EmojiOutput image={genImage} error={error} isLoading={isLoading} />
    </div>
  );
}

export default App;
