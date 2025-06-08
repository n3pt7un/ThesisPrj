# Thesis Project: Clustering F1 Driver Styles Using Telemetry Data

## 1. Project Overview

This thesis project aims to **cluster Formula 1 drivers based on their driving styles** across multiple races and seasons (2018–2024). By leveraging detailed telemetry data from the FastF1 API, the project's central goal is to develop a robust **distance metric** that quantifies stylistic similarities between drivers. This metric will facilitate effective clustering, ultimately revealing hidden patterns and nuances in driver techniques that go beyond raw performance.

The core of the project involves a sophisticated data pipeline that extracts, preprocesses, and engineers a rich set of features from telemetry data. These features are then used to build a comprehensive driver profile, which serves as the basis for clustering analysis.

## 2. Project Architecture

The project is architected with a clear separation of concerns, comprising a data analysis core built with Jupyter Notebooks and an interactive visualization dashboard powered by Streamlit.

- **`thesisprj/`**: The root directory for the analysis.
    - **Jupyter Notebooks (`*.ipynb`)**: These notebooks form the backbone of the project, handling data extraction, preprocessing, feature engineering, and clustering analysis. Key notebooks include:
        - `aggregatingTelemetry.ipynb`: Aggregates and aligns telemetry data from multiple laps and sessions.
        - `merging_CarPos.ipynb`: Merges car position data with telemetry to provide spatial context.
    - **`data_outputs/`**: Contains processed datasets, intermediate results, and cached data.

- **`thesisprj/StreamlitDashboard/`**: A self-contained Streamlit application for visualization and interactive analysis.
    - **`dashboard.py`**: The main entry point for the Streamlit app. It orchestrates the UI and user interactions.
    - **`backend.py`**: A modular backend that handles all data loading and processing logic, interacting with the FastF1 API and providing clean data to the frontend.
    - **`visualizations.py`**: Contains a suite of Plotly-based functions for rendering interactive charts and visualizations, complete with F1 team branding.

## 3. Implemented Features

- **Telemetry Extraction Pipeline**: A robust pipeline to extract, align, and merge telemetry data (Speed, RPM, Gear, Throttle, Brake, DRS) per lap and per corner. It uses **spline-fitting** to standardize distance-based metrics, ensuring accurate comparisons.
- **Advanced Data Cleaning**: Sophisticated handling of missing data (NaNs, infs) and outlier laps (crashes, anomalies) to ensure data quality.
- **Driver Style Metrics**: A comprehensive set of metrics to capture driving style, including braking points, cornering speeds, throttle application, and gear shift patterns.
- **Interactive Visualization Dashboard**: A powerful Streamlit dashboard to interactively explore telemetry data, compare driver inputs, and debug metric calculations on a lap-by-lap basis.

### Running the Dashboard

To launch the interactive visualization dashboard:
```bash
cd thesisprj/StreamlitDashboard
streamlit run dashboard.py
```

## 4. Data Pipeline In-Depth

The data pipeline is designed to efficiently process cached FastF1 API session data (`.ff1pkl` files) from 2018-2024.

1.  **Data Extraction**: The `backend.py` module fetches session data, including extended timing data, car telemetry, driver information, and session status, using the FastF1 API. It leverages an `lru_cache` to optimize performance by caching recently accessed session data.
2.  **Preprocessing & Cleaning**: Raw data is meticulously cleaned to handle common issues like missing values and outliers. Laps affected by incidents are flagged and can be excluded from analysis.
3.  **Feature Engineering**: A wide array of features is engineered from the raw telemetry. This includes calculating metrics like corner entry/exit speeds, throttle/brake application intensity, and gear shift timings.
4.  **Aggregation**: The processed data is aggregated to create a feature matrix suitable for clustering. This matrix represents a comprehensive profile of each driver's style across a season.

## 5. Feature Engineering and Normalization

The project's success hinges on engineering features that effectively capture driving style.

- **Core Metrics**:
    - **Braking**: Brake point distance from corner turn-in, braking intensity.
    - **Cornering**: Minimum cornering speed, apex location, steering angle variance.
    - **Acceleration**: Throttle application post-apex, acceleration out of corners.
    - **Gear Usage**: Gear shift patterns, timing of upshifts/downshifts.

- **Normalization Strategy**:
    - **Track Differences**: Metrics are normalized within track types (e.g., street circuits, high-speed) using **Z-scores** to enable meaningful cross-track comparisons.
    - **Car Performance**: To isolate driver skill from car performance, metrics are normalized against teammate performance or the session's best performers.

- **Missing Data Handling**: For drivers with a DNF, data from practice or qualifying laps (within 106% of the fastest lap) is used as a fallback to ensure data completeness.

## 6. Clustering and Analysis

- **Clustering Methods**: **K-Means** serves as the initial baseline, with plans to explore more advanced techniques like **Agglomerative Clustering**, **DBSCAN**, and **Spectral Clustering**.
- **Dimensionality Reduction for Visualization**: **t-SNE** and **UMAP** are employed to visualize and interpret the resulting clusters in 2D or 3D space. This is done *after* clustering to avoid information loss during preprocessing.
- **Data Structure**: The final data structure for clustering is a matrix where rows represent individual drivers and columns represent their aggregated and normalized style metrics across all races in a season.

## 7. Roadmap and Future Improvements

This section outlines the key challenges, open decisions, and next steps for the project.

- **Finalize Feature Set**: Rigorously select the most robust and style-descriptive metrics through extensive exploratory data analysis (EDA).
- **Refine Normalization Models**:
    - Solidify the rules for categorizing track types based on empirical data.
    - Evaluate and decide on a final method for normalizing out car performance effects (e.g., teammate comparison vs. a regression-based model).
- **Complete and Validate Clustering**:
    - Execute and critically evaluate the performance of different clustering algorithms.
    - Perform a deep-dive interpretation of the resulting clusters, validating them against known driver traits and expert analysis.
- **Enhance Visualization Suite**: Add more advanced features to the Streamlit dashboard for in-depth cluster exploration and comparison.
- **Thesis Write-up**: Draft the final thesis, detailing the methodology, feature engineering process, clustering results, and a thorough discussion of the conclusions.

---

### Project Status Summary

| Stage               | Status | Notes                                                                               |
| ------------------- | :----: | ----------------------------------------------------------------------------------- |
| Topic Definition    |   ✅   | Clustering F1 drivers by style using telemetry, multi-season                        |
| Data Pipeline       |   🟡   | Extraction, cleaning, aggregation in progress; need final metric set                |
| Metric Engineering  |   🟡   | Several core metrics in place, expanding/validating additional features             |
| Normalization       |   ⏳   | Track type grouping: draft approach; car normalization: method TBD                  |
| Matrix Construction |   ⏳   | Structure clear, waiting for finalized normalized metrics                           |
| Clustering          |   ⏳   | Will use K-means + alternatives, post-matrix; visualization with UMAP/t-SNE planned |
| Interpretation      |   ⏳   | Will validate against known style differences, e.g., aggressive vs. smooth drivers  |
| Writing             |   ⏳   | Will begin once feature set and matrix are finalized                                |

*(`✅ Done`, `🟡 In Progress`, `⏳ Planned`)*
