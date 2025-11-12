export type FeatureKey =
  | "radius_mean"
  | "texture_mean"
  | "perimeter_mean"
  | "area_mean"
  | "smoothness_mean"
  | "compactness_mean"
  | "concavity_mean"
  | "concave_points_mean"
  | "symmetry_mean"
  | "fractal_dimension_mean";

export interface FeatureConfig {
  key: FeatureKey;
  label: string;
  helper: string;
  placeholder: string;
  min: number;
  max: number;
  step: number;
}

export interface ReasonRow {
  feature: string;
  value: number;
  contribution: number;
  impact: string;
}

export interface PredictionResponse {
  prediction: string;
  label: number;
  probability: number | null;
  reasons: ReasonRow[];
  base_value?: number | null;
}

export interface PredictionRequest {
  [key: string]: number;
}
