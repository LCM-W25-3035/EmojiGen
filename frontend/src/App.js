// src/App.js
import React, { useState } from 'react';
import Header from './components/Header';
import EmojiForm from './components/EmojiForm';
import EmojiOutput from './components/EmojiOutput';
import axios from 'axios';
import './App.css';

function App() {
  const [images, setImages] = useState(['/smile.png', '/smile.png']);

  const handleGenerate = async (prompt, imgType) => {
    const response = await axios.post('YOUR_API_ENDPOINT', {
      prompt,
      type: imgType
    });
    const generatedImages = Array(3).fill('./smile.png'); 
    setImages(generatedImages);
  };

  return (
    <div className="App">
      <Header />
      <EmojiForm onGenerate={handleGenerate} />
      <EmojiOutput images={images} />
    </div>
  );
}

export default App;
