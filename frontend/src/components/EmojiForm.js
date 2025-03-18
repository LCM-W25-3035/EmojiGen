// src/components/EmojiForm.js
import React, { useState } from 'react';

const EmojiForm = ({ onGenerate }) => {
  const [prompt, setPrompt] = useState('');
  const [imgType, setImgType] = useState('Emoji');

  const handleSubmit = (e) => {
    e.preventDefault();
    onGenerate(prompt, imgType);
  };

  return (
    <form className="user-input" onSubmit={handleSubmit}>
      <div className="prompt">
        <input
          type="text"
          placeholder="Grinning face with sunglasses"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          required
        />
        <div className="img-type-select">
          <input
            type="radio"
            name="img_type"
            id="img-emoji"
            value="Emoji"
            checked={imgType === 'Emoji'}
            onChange={(e) => setImgType(e.target.value)}
          />
          <label htmlFor="img-emoji">Emoji</label>

          <input
            type="radio"
            name="img_type"
            id="img-sticker"
            value="Sticker"
            checked={imgType === 'Sticker'}
            onChange={(e) => setImgType(e.target.value)}
          />
          <label htmlFor="img-sticker">Sticker</label>
        </div>
      </div>
      <button className="gen-btn" type="submit">Generate</button>
    </form>
  );
};

export default EmojiForm;
