// src/components/EmojiOutput.js
import React from 'react';

const EmojiOutput = ({ images }) => (
  <main className="output">
    {images.map((img, index) => (
      <div className="img-blocks" key={index}>
        <img src={img} alt={`emoji ${index}`} />
      </div>
    ))}
  </main>
);

export default EmojiOutput;
