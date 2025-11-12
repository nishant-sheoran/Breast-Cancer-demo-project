import { FeatureConfig } from "../types/prediction";

export const featureConfigs: FeatureConfig[] = [
  {
    key: "radius_mean",
    label: "Mean radius (mm)",
    helper: "Typical range 6 - 28",
    placeholder: "14.0",
    min: 5,
    max: 35,
    step: 0.1
  },
  {
    key: "texture_mean",
    label: "Mean texture",
    helper: "Typical range 9 - 40",
    placeholder: "19.0",
    min: 5,
    max: 45,
    step: 0.1
  },
  {
    key: "perimeter_mean",
    label: "Mean perimeter (mm)",
    helper: "Typical range 45 - 190",
    placeholder: "90.0",
    min: 20,
    max: 250,
    step: 0.1
  },
  {
    key: "area_mean",
    label: "Mean area (mm^2)",
    helper: "Typical range 150 - 2500",
    placeholder: "700",
    min: 100,
    max: 4000,
    step: 1
  },
  {
    key: "smoothness_mean",
    label: "Mean smoothness",
    helper: "Typical range 0.05 - 0.2",
    placeholder: "0.1",
    min: 0.02,
    max: 0.3,
    step: 0.001
  },
  {
    key: "compactness_mean",
    label: "Mean compactness",
    helper: "Typical range 0.02 - 0.35",
    placeholder: "0.15",
    min: 0.0,
    max: 0.5,
    step: 0.001
  },
  {
    key: "concavity_mean",
    label: "Mean concavity",
    helper: "Typical range 0.0 - 0.45",
    placeholder: "0.2",
    min: 0.0,
    max: 0.7,
    step: 0.001
  },
  {
    key: "concave_points_mean",
    label: "Mean concave points",
    helper: "Typical range 0.0 - 0.2",
    placeholder: "0.09",
    min: 0.0,
    max: 0.3,
    step: 0.001
  },
  {
    key: "symmetry_mean",
    label: "Mean symmetry",
    helper: "Typical range 0.1 - 0.3",
    placeholder: "0.2",
    min: 0.05,
    max: 0.4,
    step: 0.001
  },
  {
    key: "fractal_dimension_mean",
    label: "Mean fractal dimension",
    helper: "Typical range 0.04 - 0.1",
    placeholder: "0.06",
    min: 0.02,
    max: 0.2,
    step: 0.001
  }
];
