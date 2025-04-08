import React, { useState } from 'react';

const EmojiForm = ({ onGenerate, isLoading }) => {
  const [prompt, setPrompt] = useState('');
  const [imgType, setImgType] = useState('Emoji');

  const handleSubmit = (e) => {
    e.preventDefault();
    onGenerate(prompt, imgType);
  };

  return (
    <form className="user-input" onSubmit={handleSubmit}>
      <div className="prompt">
        <input type="text"
          placeholder="Grinning face with sunglasses"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          required
        />
        <div className="img-type-select">
          <div>
            <input
              type="radio"
              name="img_type"
              id="img-emoji"
              value="Emoji"
              checked={imgType === 'Emoji'}
              onChange={(e) => setImgType(e.target.value)}
            />
            &nbsp;
            <label htmlFor="img-emoji">Emoji</label>
          </div>

          <div>
            <input
              type="radio"
              name="img_type"
              id="img-sticker"
              value="Sticker"
              checked={imgType === 'Sticker'}
              onChange={(e) => setImgType(e.target.value)}
            />
            &nbsp;
            <label htmlFor="img-sticker">Sticker</label>
          </div>
        </div>
      </div>
      <button className="gen-btn" type="submit">{isLoading ? 'Loading..' : 'Generate'}</button>
    </form>
  );
};

export default EmojiForm;
