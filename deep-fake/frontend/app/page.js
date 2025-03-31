"use client";
import { useEffect, useRef, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Label } from "@/components/ui/label";
import {
  AlertCircle,
  Camera,
  FileUp,
  Loader2,
  CheckCircle,
  XCircle,
} from "lucide-react";

export default function Home() {
  // State management
  const [step, setStep] = useState("intro"); // 'intro', 'upload', 'camera', 'processing', 'result'
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [processingProgress, setProcessingProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  // Refs for media handling
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // Setup webcam when camera step is activated
  useEffect(() => {
    if (step === "camera") {
      setupCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [step]);

  // Setup camera function
  const setupCamera = async () => {
    setCameraError("");
    setCameraActive(false); // Reset camera active state before starting

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Browser doesn't support camera access");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 720 },
          height: { ideal: 480 },
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        // The issue is likely here - we need to ensure the video element is properly handling loadedmetadata
        videoRef.current.onloadedmetadata = () => {
          // Ensure we're using a Promise and properly handling it
          videoRef.current
            .play()
            .then(() => {
              console.log("Camera successfully started");
              setCameraActive(true); // Set camera active only after successful play
            })
            .catch((e) => {
              console.error("Error playing video:", e);
              setCameraError("Could not start video playback: " + e.message);
            });
        };
      } else {
        throw new Error("Video reference is not available");
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      setCameraError(
        err.message || "Could not access your camera. Please check permissions."
      );
    }
  };

  // Stop camera function
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setCameraActive(false);
    }
  };

  // Handle file selection
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadedImage(URL.createObjectURL(file));
    }
  };

  // Send to backend function
  const sendToBackend = async (base64String) => {
    setIsProcessing(true);
    setStep("processing");

    // Simulate progress for better UX
    const progressInterval = setInterval(() => {
      setProcessingProgress((prev) => {
        const newValue = prev + Math.random() * 10;
        return newValue > 90 ? 90 : newValue; // Cap at 90% until we get actual results
      });
    }, 300);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: base64String }),
      });

      if (!response.ok) throw new Error("Backend error");

      const data = await response.json();
      setPrediction(data.prediction);
      setConfidence((data.confidence * 100).toFixed(2) + "%");

      // Set progress to 100% when done
      setProcessingProgress(100);
      setTimeout(() => setStep("result"), 500);
    } catch (err) {
      console.error(err);
      alert("Something went wrong: " + err.message);
      setStep("intro");
    } finally {
      clearInterval(progressInterval);
      setIsProcessing(false);
    }
  };

  // Handle predict function
  const handlePredict = () => {
    if (!selectedFile) {
      alert("Please select an image first!");
      return;
    }

    const reader = new FileReader();
    reader.onload = async (event) => {
      const base64String = event.target.result.split(",")[1];
      await sendToBackend(base64String);
    };
    reader.readAsDataURL(selectedFile);
  };

  // Capture from camera
  const captureFromCamera = async () => {
    if (!cameraActive) {
      alert("Camera is not active. Please allow camera access.");
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    // Set canvas dimensions to match the video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the current video frame onto the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to data URL and extract base64 part
    const imageDataUrl = canvas.toDataURL("image/jpeg");
    setUploadedImage(imageDataUrl);

    const base64Image = imageDataUrl.split(",")[1];
    await sendToBackend(base64Image);
  };

  // Reset function for starting over
  const resetDetector = () => {
    setStep("intro");
    setPrediction("");
    setConfidence("");
    setSelectedFile(null);
    setUploadedImage(null);
    setProcessingProgress(0);
    setCameraError("");
  };

  // Mock prediction function for testing when backend is not available
  const mockPredict = () => {
    setIsProcessing(true);
    setStep("processing");

    const progressInterval = setInterval(() => {
      setProcessingProgress((prev) => {
        const newValue = prev + Math.random() * 10;
        return newValue >= 100 ? 100 : newValue;
      });
    }, 300);

    setTimeout(() => {
      clearInterval(progressInterval);
      setProcessingProgress(100);

      const isFake = Math.random() > 0.5;
      setPrediction(isFake ? "Fake" : "Real");
      setConfidence((Math.random() * 30 + 70).toFixed(2) + "%");

      setTimeout(() => {
        setStep("result");
        setIsProcessing(false);
      }, 500);
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white flex items-center justify-center p-4">
      <Card className="w-full max-w-md bg-gray-800 border-gray-700 shadow-xl overflow-hidden">
        {/* Header */}
        <CardHeader className="bg-gray-900 border-b border-gray-700">
          <CardTitle className="text-center text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
            Deepfake Detector
          </CardTitle>
          <CardDescription className="text-center text-gray-400">
            Uncover the truth behind the image
          </CardDescription>
        </CardHeader>

        <CardContent className="p-6">
          {/* Intro Step */}
          {step === "intro" && (
            <div className="space-y-6">
              <p className="text-gray-300 text-center mb-6">
                Select how you want to check for deepfakes
              </p>

              <div className="grid grid-cols-2 gap-4">
                <Button
                  variant="outline"
                  className="h-32 flex flex-col items-center justify-center space-y-2 border border-gray-700 bg-gray-800 hover:bg-gray-700 hover:border-blue-500 transition-all"
                  onClick={() => setStep("upload")}
                >
                  <FileUp size={32} className="text-blue-400" />
                  <span>Upload Image</span>
                </Button>

                <Button
                  variant="outline"
                  className="h-32 flex flex-col items-center justify-center space-y-2 border border-gray-700 bg-gray-800 hover:bg-gray-700 hover:border-purple-500 transition-all"
                  onClick={() => setStep("camera")}
                >
                  <Camera size={32} className="text-purple-400" />
                  <span>Use Camera</span>
                </Button>
              </div>
            </div>
          )}

          {/* Upload Step */}
          {step === "upload" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-4">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setStep("intro")}
                  className="text-gray-400 hover:text-white"
                >
                  Back
                </Button>
                <h3 className="text-lg font-medium">Upload an Image</h3>
              </div>

              <div
                className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors flex flex-col items-center justify-center space-y-4"
                onClick={() => fileInputRef.current.click()}
              >
                {uploadedImage ? (
                  <div className="w-full">
                    <img
                      src={uploadedImage}
                      alt="Preview"
                      className="max-h-48 max-w-full mx-auto rounded object-contain"
                    />
                    <p className="text-gray-400 text-sm mt-2">
                      Click to change image
                    </p>
                  </div>
                ) : (
                  <>
                    <FileUp size={48} className="text-gray-500" />
                    <div>
                      <p className="text-gray-300">
                        Drag and drop your image here
                      </p>
                      <p className="text-gray-500 text-sm">
                        or click to browse files
                      </p>
                    </div>
                  </>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </div>

              <div className="flex space-x-4">
                <Button
                  className="flex-1 bg-blue-600 hover:bg-blue-700"
                  disabled={!selectedFile || isProcessing}
                  onClick={handlePredict}
                >
                  Analyze with API
                </Button>

                <Button
                  className="flex-1 bg-purple-600 hover:bg-purple-700"
                  disabled={!selectedFile || isProcessing}
                  onClick={mockPredict}
                >
                  Demo Mode
                </Button>
              </div>
            </div>
          )}

          {/* Camera Step */}
          {step === "camera" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between mb-4">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setStep("intro")}
                  className="text-gray-400 hover:text-white"
                >
                  Back
                </Button>
                <h3 className="text-lg font-medium">Camera Capture</h3>
              </div>

              <div className="relative rounded-lg overflow-hidden bg-black aspect-video flex items-center justify-center">
                {cameraError ? (
                  <div className="text-red-500 flex flex-col items-center space-y-2 p-4 text-center">
                    <AlertCircle size={32} />
                    <span>Camera error: {cameraError}</span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={setupCamera}
                      className="mt-2"
                    >
                      Try Again
                    </Button>
                  </div>
                ) : (
                  // Always show the video element, but add a loading state on top if not active
                  <div className="w-full h-full relative">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover"
                    />

                    {!cameraActive && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/70">
                        <Loader2
                          size={32}
                          className="animate-spin text-blue-400"
                        />
                        <span className="mt-2 text-gray-300">
                          Initializing camera...
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="flex space-x-4">
                <Button
                  className="flex-1 bg-purple-600 hover:bg-purple-700"
                  disabled={!cameraActive || isProcessing}
                  onClick={captureFromCamera}
                >
                  <Camera size={16} className="mr-2" />
                  Capture & Analyze
                </Button>

                <Button
                  className="flex-1 bg-blue-600 hover:bg-blue-700"
                  disabled={isProcessing}
                  onClick={mockPredict}
                >
                  Demo Mode
                </Button>
              </div>

              {/* Hidden canvas for capturing image */}
              <canvas
                ref={canvasRef}
                width="640"
                height="480"
                className="hidden"
              />
            </div>
          )}

          {/* Processing Step */}
          {step === "processing" && (
            <div className="space-y-6 py-4">
              <div className="flex flex-col items-center justify-center space-y-4">
                <div className="w-24 h-24 rounded-full bg-gray-700 flex items-center justify-center">
                  <Loader2 size={32} className="text-blue-400 animate-spin" />
                </div>
                <h3 className="text-xl font-medium">Analyzing Image</h3>
                <p className="text-gray-400 text-center">
                  Our AI is detecting whether this image is genuine or a
                  deepfake
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm text-gray-400">
                  <span>Processing</span>
                  <span>{Math.round(processingProgress)}%</span>
                </div>
                <Progress
                  value={processingProgress}
                  className="h-2 bg-gray-700"
                />
              </div>
            </div>
          )}

          {/* Result Step */}
          {step === "result" && (
            <div className="space-y-6">
              <div className="flex items-center justify-center">
                {uploadedImage && (
                  <div className="relative rounded-lg overflow-hidden w-full h-48 mb-4">
                    <img
                      src={uploadedImage}
                      alt="Analyzed"
                      className="w-full h-full object-cover"
                    />
                    <div
                      className={`absolute inset-0 flex items-center justify-center ${
                        prediction.toLowerCase() === "real"
                          ? "bg-green-900/30"
                          : "bg-red-900/30"
                      }`}
                    >
                      {prediction.toLowerCase() === "real" ? (
                        <CheckCircle
                          size={64}
                          className="text-green-400 opacity-80"
                        />
                      ) : (
                        <XCircle
                          size={64}
                          className="text-red-400 opacity-80"
                        />
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div
                className={`rounded-lg p-4 ${
                  prediction.toLowerCase() === "real"
                    ? "bg-green-900/20 border border-green-800"
                    : "bg-red-900/20 border border-red-800"
                }`}
              >
                <div className="flex items-center space-x-4">
                  {prediction.toLowerCase() === "real" ? (
                    <CheckCircle className="text-green-400 shrink-0" />
                  ) : (
                    <AlertCircle className="text-red-400 shrink-0" />
                  )}
                  <div>
                    <h3 className="font-medium text-lg">
                      {prediction.toLowerCase() === "real"
                        ? "Authentic Image Detected"
                        : "Deepfake Detected"}
                    </h3>
                    <p className="text-gray-400 text-sm">
                      {prediction.toLowerCase() === "real"
                        ? "Our analysis indicates this is likely a genuine image."
                        : "Our analysis indicates this image has likely been manipulated."}
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 rounded-lg p-4">
                <Label className="text-gray-400 text-sm">
                  Confidence Level
                </Label>
                <div className="flex items-center justify-between mt-1">
                  <span className="text-xl font-bold">{confidence}</span>
                  <Progress
                    value={parseFloat(confidence)}
                    className="h-2 w-32 bg-gray-700"
                  />
                </div>
              </div>
            </div>
          )}
        </CardContent>

        {/* Footer */}
        <CardFooter
          className={`bg-gray-900 border-t border-gray-700 p-4 ${
            step !== "result" ? "hidden" : ""
          }`}
        >
          <div className="w-full space-y-2">
            <Button
              className="w-full bg-blue-600 hover:bg-blue-700"
              onClick={resetDetector}
            >
              Check Another Image
            </Button>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}
