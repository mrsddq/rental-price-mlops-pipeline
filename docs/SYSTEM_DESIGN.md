# System Design

This repository is a compact MLOps case study: one dataset, one baseline model, one quality gate, one compiled pipeline, and one lightweight serving path.

```mermaid
flowchart LR
    A["Versioned sample data"] --> B["Schema validation"]
    B --> C["Feature/target split"]
    C --> D["Local model training"]
    C --> E["Mean-price baseline"]
    D --> F["Metrics: RMSE, MAE, R2"]
    E --> F
    F --> G["Quality gate"]
    G --> H["Model card + quality report"]
    G --> I["Model artifact"]
    D --> J["Kubeflow pipeline YAML"]
    I --> K["FastAPI serving layer"]
    K --> L["Prediction event with range warnings"]
```

## Request Path

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI service
    participant Model as Fitted model
    participant Monitor as Input monitor

    Client->>API: POST /predict {rooms, sqft}
    API->>Model: predict feature row
    Model-->>API: predicted price
    API->>Monitor: compare input with training ranges
    Monitor-->>API: warnings
    API-->>Client: price + event metadata
```

## Production Gaps

- Replace local pickle artifacts with a model registry.
- Add request logging to durable storage.
- Track prediction/label joins after ground truth arrives.
- Add drift dashboards for feature ranges and error metrics.
- Split the Kubeflow pipeline into data validation, training, evaluation, registration, and promotion components.
