import { PredictionResponse } from "../types/prediction";

interface Props {
  result: PredictionResponse | null;
  error: string | null;
}

export function PredictionPanel({ result, error }: Props) {
  if (error) {
    return <div className="card error">{error}</div>;
  }

  if (!result) {
    return <div className="card muted">Submit patient metrics to see the model output.</div>;
  }

  return (
    <div className="card">
      <h2>Prediction</h2>
      <p className={`badge ${result.label === 1 ? "malignant" : "benign"}`}>{result.prediction}</p>
      {typeof result.probability === "number" && (
        <p className="probability">
          Malignancy probability: <strong>{(result.probability * 100).toFixed(1)}%</strong>
        </p>
      )}
      <h3>Reasoning</h3>
      <p>The top signals driving this decision are listed below.</p>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Value</th>
              <th>Impact</th>
              <th>Contribution</th>
            </tr>
          </thead>
          <tbody>
            {result.reasons.map((reason) => (
              <tr key={reason.feature}>
                <td>{reason.feature}</td>
                <td>{reason.value.toFixed(4)}</td>
                <td className={reason.impact.includes("malignant") ? "malignant" : "benign"}>
                  {reason.impact.replace("_", " ")}
                </td>
                <td>{reason.contribution.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
