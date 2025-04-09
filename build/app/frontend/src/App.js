import React, { useState } from 'react';
import Header from './components/Header';
import EmojiForm from './components/EmojiForm';
import EmojiOutput from './components/EmojiOutput';
import axios from 'axios';
import './App.css';

// For React (Create React App)
const baseURL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const axiosInstance = axios.create({
  baseURL, // All requests using this instance use your environment-specific URL
  // Additional configuration options such as headers can be added here.
});

function App() {
  const [genImage, setGenImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async (prompt, imgType, genModel) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await axiosInstance.post('/generate', {
        "prompt": prompt,
        "gen_model": genModel,
        "image_type": imgType
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
