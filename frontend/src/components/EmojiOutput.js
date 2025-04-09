import React from 'react';

const EmojiOutput = ({ image, error, isLoading }) => {
  return (

    <main className="output">
      {isLoading ?
        <div className="flex-center">
          <div className="loader" />
          <div className='load-text'>We really hope it works...</div>
        </div> :
        error ? <div className="alert alert-danger">
          <div className="flex-center">
            <div className='load-text'>{error}</div>
          </div>
        </div> :
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
