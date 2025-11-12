import { FormEvent, useState } from "react";
import { featureConfigs } from "../lib/features";
import { PredictionRequest } from "../types/prediction";

interface Props {
  onSubmit: (payload: PredictionRequest) => void;
  loading: boolean;
}

const buildInitialState = () => {
  return featureConfigs.reduce<Record<string, number>>((acc, feature) => {
    acc[feature.key] = Number(feature.placeholder);
    return acc;
  }, {});
};

export function FeatureForm({ onSubmit, loading }: Props) {
  const [values, setValues] = useState<Record<string, number>>(buildInitialState);

  const handleChange = (key: string, value: number) => {
    setValues((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    onSubmit(values);
  };

  return (
    <form onSubmit={handleSubmit} className="feature-form">
      <div className="grid">
        {featureConfigs.map((config) => (
          <label key={config.key} className="input-block">
            <span className="input-label">{config.label}</span>
            <input
              type="number"
              value={values[config.key] ?? ""}
              min={config.min}
              max={config.max}
              step={config.step}
              placeholder={config.placeholder}
              onChange={(event) => handleChange(config.key, Number(event.target.value))}
              required
            />
            <small>{config.helper}</small>
          </label>
        ))}
      </div>
      <button type="submit" disabled={loading}>
        {loading ? "Predicting..." : "Predict"}
      </button>
    </form>
  );
}
