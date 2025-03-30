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

  const handleGenerate = async (prompt) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/generate-from-text', {
        text: prompt
      });

      setGenImage(response.data.image);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate emoji');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <Header />
      <EmojiForm onGenerate={handleGenerate} isLoading={isLoading} />
      <EmojiOutput image={genImage} error={error} isLoading={isLoading} />
    </div>
  );
}

export default App;
