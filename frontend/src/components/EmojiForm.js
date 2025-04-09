import React, { useState } from 'react';

const EmojiForm = ({ onGenerate, isLoading }) => {
  const [prompt, setPrompt] = useState('');
  const [imgType, setImgType] = useState('Emoji');
  const [genModel, setGenModel] = useState('')

  const handleSelectChange = (event) => {
    setGenModel(event.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onGenerate(prompt, imgType, genModel);
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
      <div>
        <select
          className="gen-model"
          value={genModel}
          onChange={handleSelectChange}
        >
          {/* Provide an explicit value for each option */}
          <option value="">Select Model</option>
          <option value="gan">GANs</option>
          <option value="diffusion">Diffusion</option>
        </select>
      </div>
      {isLoading ? <button className="gen-btn" disabled>generating..</button>
      : <button className="gen-btn" type="submit">Generate</button>}
      
    </form>
  );
};

export default EmojiForm;
