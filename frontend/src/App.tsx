import { useState } from "react";
import axios from "axios";
import api from "./api/client";
import { FeatureForm } from "./components/FeatureForm";
import { PredictionPanel } from "./components/PredictionPanel";
import { PredictionRequest, PredictionResponse } from "./types/prediction";

function App() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (payload: PredictionRequest) => {
    setPending(true);
    setError(null);
    try {
      const response = await api.post<PredictionResponse>("/predict", payload);
      setResult(response.data);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(typeof detail === "string" ? detail : err.message);
      } else {
        setError("Prediction failed");
      }
      setResult(null);
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="page">
      <header>
        <div>
          <p className="eyebrow">Breast Cancer Toolkit</p>
          <h1>Prediction Dashboard</h1>
          <p className="subtitle">
            Provide the ten diagnostic measurements to receive the model output plus an explanation of the driving
            factors.
          </p>
        </div>
      </header>
      <main>
        <section className="column">
          <FeatureForm onSubmit={handleSubmit} loading={pending} />
        </section>
        <section className="column">
          <PredictionPanel result={result} error={error} />
        </section>
      </main>
      <footer>
        <small>Note: This interface is for research support only and is not a medical diagnostic tool.</small>
      </footer>
    </div>
  );
}

export default App;
