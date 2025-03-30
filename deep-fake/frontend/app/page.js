"use client";
import { useState } from "react";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      alert("Please select an image first!");
      return;
    }

    // Convert the image file to base64
    const reader = new FileReader();
    reader.onload = async (event) => {
      const base64String = event.target.result.split(",")[1];

      try {
        // Send it to FastAPI's /predict endpoint
        const response = await fetch("http://localhost:8000/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image_b64: base64String,
          }),
        });

        if (!response.ok) {
          throw new Error("Error from backend: " + response.statusText);
        }

        const data = await response.json();
        setPrediction(data.prediction);
        setConfidence((data.confidence * 100).toFixed(2) + "%");
      } catch (err) {
        console.error(err);
        alert("Something went wrong: " + err.message);
      }
    };

    reader.readAsDataURL(selectedFile);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "1rem",
        maxWidth: "500px",
        margin: "auto",
        padding: "2rem",
      }}
    >
      <h1>Deepfake Detector</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handlePredict}>Check Image</button>

      {prediction && (
        <div>
          <h2>Result</h2>
          <p>
            Prediction: <b>{prediction}</b>
          </p>
          <p>
            Confidence: <b>{confidence}</b>
          </p>
        </div>
      )}
    </div>
  );
}
