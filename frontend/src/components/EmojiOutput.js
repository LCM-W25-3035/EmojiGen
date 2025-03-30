// src/components/EmojiOutput.js
import React from 'react';

const EmojiOutput = ({ image, error, isLoading }) => {
  return (

    <main className="output">
      {isLoading ? <div className="text-center my-4">Generating...</div> :
        error ? <div className="alert alert-danger">{error}</div> :
          !image ? <div className="text-center my-4">Enter a prompt!</div> :
            <div className="img-blocks">
              <img
                src={`data:image/png;base64,${image}`}
                alt="Generated emoji"
              />
            </div>
      }
    </main>
  );
};
export default EmojiOutput;
