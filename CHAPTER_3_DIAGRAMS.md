# Chapter 3: Analysis & Design Diagrams
## Nairobi Urban Flood Digital Twin

---

# 3.5 ANALYSIS DIAGRAMS

## 3.5.1 Use Case Diagram

```mermaid
graph TB
    subgraph System["Nairobi Flood Digital Twin Platform"]
        Dashboard["🗺️ Dashboard System"]
    end
    
    subgraph Actors["External Actors"]
        Planner["👔 Municipal Planner"]
        Responder["🚨 Emergency Responder"]
        Admin["⚙️ System Administrator"]
        Citizen["👤 Citizen/Public"]
    end
    
    subgraph UseCases["Use Cases"]
        UC1["View Real-time<br/>Flood Risk Map"]
        UC2["Adjust Rainfall<br/>Scenario Sliders"]
        UC3["View 3D Flood<br/>Projections"]
        UC4["Retrieve Localized<br/>Hazard Summaries"]
        UC5["Generate<br/>Emergency Reports"]
        UC6["Manage User<br/>Accounts"]
        UC7["Update Sensor<br/>Data Streams"]
        UC8["View Flood<br/>Forecast"]
    end
    
    Planner -->|uses| UC2
    Planner -->|uses| UC3
    Planner -->|uses| UC5
    
    Responder -->|uses| UC1
    Responder -->|uses| UC3
    Responder -->|uses| UC4
    Responder -->|uses| UC5
    
    Admin -->|manages| UC6
    Admin -->|manages| UC7
    
    Citizen -->|views| UC1
    Citizen -->|views| UC8
    
    UC1 -->|part of| Dashboard
    UC2 -->|part of| Dashboard
    UC3 -->|part of| Dashboard
    UC4 -->|part of| Dashboard
    UC5 -->|part of| Dashboard
    UC6 -->|part of| Dashboard
    UC7 -->|part of| Dashboard
    UC8 -->|part of| Dashboard
    
    style Dashboard fill:#e3f2fd
    style Planner fill:#fff3e0
    style Responder fill:#ffebee
    style Admin fill:#f3e5f5
    style Citizen fill:#e8f5e9
```

---

## 3.5.2 Sequence Diagram

```mermaid
sequenceDiagram
    participant User as User<br/>(Responder)
    participant UI as Dashboard UI<br/>(Plotly Dash)
    participant API as Backend API<br/>(Flask)
    participant Model as U-Net Model<br/>Prediction Engine
    participant DB as Database<br/>(PostGIS)
    participant Render as WebGL<br/>3D Renderer
    
    User->>UI: 1. Set Rainfall Parameter<br/>(e.g., 50mm in 7 days)
    activate UI
    
    UI->>API: 2. POST /predict<br/>with rainfall + location
    activate API
    
    API->>DB: 3. Query Terrain Data<br/>(HAND, elevation, built-up)
    activate DB
    DB-->>API: 4. Return spatial features
    deactivate DB
    
    API->>Model: 5. Forward Pass<br/>SAR + Terrain + Rainfall
    activate Model
    Model-->>API: 6. Return Flood Probability Map<br/>(0-1 per pixel)
    deactivate Model
    
    API->>DB: 7. Cache Results<br/>Simulation_ID, timestamp
    activate DB
    DB-->>API: 8. Confirm stored
    deactivate DB
    
    API-->>UI: 9. Return JSON<br/>flood_probability, bounds
    deactivate API
    
    UI->>Render: 10. Render 3D Overlay<br/>WebGL + Pydeck
    activate Render
    Render-->>UI: 11. Render complete
    deactivate Render
    
    UI-->>User: 12. Display Interactive<br/>Flood Depth Map<br/>(color-coded by probability)
    deactivate UI
    
    User->>UI: 13. Click on Region<br/>Get Hazard Summary
    activate UI
    UI->>API: 14. GET /hazard_summary<br/>?location=region
    activate API
    API-->>UI: 15. Return Statistics<br/>(affected_pop, max_depth)
    deactivate API
    UI-->>User: 16. Show Summary Panel<br/>Risk metrics + evacuation routes
    deactivate UI
```

---

## 3.5.3 Entity-Relationship Diagram (ERD)

```mermaid
erDiagram
    USERS ||--o{ SIMULATIONS : creates
    SIMULATIONS ||--o{ RESULTS : generates
    RESULTS ||--o{ FLOOD_PIXELS : contains
    
    METEOROLOGICAL_FEEDS ||--o{ SIMULATIONS : inputs
    TOPOGRAPHIC_LAYERS ||--o{ FLOOD_PIXELS : references
    SENSOR_DATA ||--o{ METEOROLOGICAL_FEEDS : feeds
    
    USERS {
        int user_id PK
        string username UK
        string email UK
        string role "admin|responder|planner"
        timestamp created_at
        timestamp last_login
    }
    
    METEOROLOGICAL_FEEDS {
        int feed_id PK
        string sensor_type "rainfall|temperature|humidity"
        float latitude
        float longitude
        float value
        string unit "mm|celsius|percent"
        timestamp observation_time
    }
    
    TOPOGRAPHIC_LAYERS {
        int layer_id PK
        string layer_name "HAND|elevation|slope|built-up"
        float latitude
        float longitude
        float cell_value
        int grid_x
        int grid_y
        geometry geom
    }
    
    SENSOR_DATA {
        int sensor_id PK
        string sensor_name
        float latitude
        float longitude
        string status "active|inactive"
        timestamp last_update
    }
    
    SIMULATIONS {
        int simulation_id PK
        int user_id FK
        int feed_id FK
        timestamp created_at
        float rainfall_mm
        string model_version
        string status "pending|running|complete"
    }
    
    RESULTS {
        int result_id PK
        int simulation_id FK
        timestamp computed_at
        float avg_flood_probability
        int pixels_flooded
        string result_path "s3://results/..."
    }
    
    FLOOD_PIXELS {
        int pixel_id PK
        int result_id FK
        int layer_id FK
        int grid_x
        int grid_y
        float flood_probability
        float elevation
        float hand_value
        float built_up_fraction
        geometry geom
    }
```

---

## 3.5.4 Class Diagram

```mermaid
classDiagram
    class GeospatialRenderer {
        -canvas: WebGLContext
        -data_layer: GeoJSON
        -color_scale: ColorMap
        +render_flood_map(): void
        +update_overlay(): void
        +add_marker(): void
        +zoom_to_bounds(): void
    }
    
    class SurrogateModel {
        -model_path: str
        -device: torch.device
        -model: UNet
        +load_weights(): void
        +forward_pass(): np.ndarray
        +predict_flood_probability(): np.ndarray
        +batch_predict(): list
    }
    
    class TelemetryData {
        -rainfall_feed: dict
        -sensor_cache: dict
        -last_update: datetime
        +fetch_rainfall(): float
        +get_sensor_reading(): float
        +validate_data(): bool
        +cache_result(): void
    }
    
    class DashboardApp {
        -app: dash.Dash
        -layout: html.Div
        -callbacks: dict
        +register_callback(): void
        +run_server(): void
        +update_map(): void
    }
    
    class DatabaseManager {
        -connection: psycopg2.connection
        -postgis_enabled: bool
        +query_topography(): dict
        +store_result(): void
        +get_user_simulations(): list
        +spatial_query(): GeoDataFrame
    }
    
    class FloodPredictor {
        -model: SurrogateModel
        -telemetry: TelemetryData
        -db: DatabaseManager
        +predict(): np.ndarray
        +postprocess_results(): dict
        +validate_spatial_plausibility(): bool
    }
    
    class APIGateway {
        -flask_app: Flask
        -predictor: FloodPredictor
        -auth_manager: AuthManager
        +post_predict(): JSON
        +get_hazard_summary(): JSON
        +authenticate_user(): bool
    }
    
    class AuthManager {
        -users_db: dict
        -session_cache: dict
        +verify_credentials(): bool
        +create_session(): str
        +validate_token(): bool
        +revoke_session(): void
    }
    
    %% Relationships
    DashboardApp --> GeospatialRenderer
    DashboardApp --> APIGateway
    APIGateway --> FloodPredictor
    FloodPredictor --> SurrogateModel
    FloodPredictor --> TelemetryData
    FloodPredictor --> DatabaseManager
    APIGateway --> AuthManager
    DatabaseManager --> TelemetryData
    GeospatialRenderer --> DatabaseManager
```

---

## 3.5.5 Activity Diagram

```mermaid
graph TD
    START([Emergency Response Portal<br/>Login]) 
    
    LOGIN["1. User Enters<br/>Username & Password"]
    AUTH{2. Credentials<br/>Valid?}
    INVALID["❌ Invalid<br/>Show Error"]
    RETRY["3. Retry Login"]
    
    DASHBOARD["4. Load Dashboard<br/>Display Nairobi Map"]
    SELECT["5. Select Catchment<br/>or Region of Interest"]
    
    PARAM["6. View Current<br/>Sensor Data"]
    ADJUST["7. Adjust Rainfall<br/>Scenario Sliders<br/>(0-200mm range)"]
    
    REQUEST["8. Trigger Prediction<br/>Submit Rainfall Input"]
    PREDICT["9. Run U-Net Model<br/>Compute Flood Probability<br/>~2 sec processing"]
    
    RENDER["10. Render 3D<br/>WebGL Visualization<br/>Color-coded Flood Map"]
    
    VIEW_RESULTS["11. Display Results<br/>- Flood Probability Grid<br/>- Affected Population<br/>- Critical Infrastructure"]
    
    HAZARD["12. Generate<br/>Hazard Summary Report<br/>(PDF/JSON)"]
    
    QUERY["13. Query Specific<br/>Zones"]
    STATS{14. Need More<br/>Information?}
    
    EXPORT["15. Export Results<br/>GeoJSON/Shapefile"]
    
    END_SUCCESS["16. ✅ Simulation Complete<br/>Share with Team"]
    END_NEW["17. Create New Scenario<br/>or Exit"]
    
    START --> LOGIN
    LOGIN --> AUTH
    AUTH -->|No| INVALID
    INVALID --> RETRY
    RETRY --> LOGIN
    AUTH -->|Yes| DASHBOARD
    
    DASHBOARD --> SELECT
    SELECT --> PARAM
    PARAM --> ADJUST
    ADJUST --> REQUEST
    
    REQUEST --> PREDICT
    PREDICT --> RENDER
    RENDER --> VIEW_RESULTS
    
    VIEW_RESULTS --> HAZARD
    HAZARD --> QUERY
    QUERY --> STATS
    
    STATS -->|Yes| QUERY
    STATS -->|No| EXPORT
    EXPORT --> END_SUCCESS
    END_SUCCESS --> END_NEW
    
    style START fill:#c8e6c9
    style LOGIN fill:#bbdefb
    style AUTH fill:#fff9c4
    style PREDICT fill:#ffe0b2
    style RENDER fill:#f8bbd0
    style END_SUCCESS fill:#c8e6c9
    style INVALID fill:#ffcdd2
```

---

# 3.6 DESIGN DIAGRAMS

## 3.6.1 Database Schema

```mermaid
graph TB
    subgraph PostGIS["PostGIS Spatial Database"]
        
        subgraph UserMgmt["User Management Tables"]
            USERS["users<br/>---<br/>user_id (PK)<br/>username<br/>email<br/>password_hash<br/>role<br/>created_at"]
            
            SESSIONS["user_sessions<br/>---<br/>session_id (PK)<br/>user_id (FK)<br/>token<br/>ip_address<br/>expires_at"]
        end
        
        subgraph SpatialData["Geospatial Data Tables"]
            TOPO["topographic_grid<br/>---<br/>cell_id (PK)<br/>grid_x, grid_y<br/>elevation (m)<br/>hand (m)<br/>slope (%)<br/>twi (index)<br/>built_up (%)<br/>geom (geometry)<br/>INDEX: spatial"]
            
            PWATER["permanent_water<br/>---<br/>water_id (PK)<br/>geom (geometry)<br/>area (km²)<br/>INDEX: spatial"]
            
            INFRA["critical_infrastructure<br/>---<br/>infra_id (PK)<br/>name<br/>type<br/>latitude<br/>longitude<br/>geom (geometry)<br/>INDEX: spatial"]
        end
        
        subgraph SensorData["Real-Time Sensor Data"]
            RAINFALL["rainfall_observations<br/>---<br/>obs_id (PK)<br/>station_id (FK)<br/>observation_time<br/>rainfall_mm<br/>latitude, longitude"]
            
            STATIONS["rainfall_stations<br/>---<br/>station_id (PK)<br/>station_name<br/>latitude<br/>longitude<br/>geom (geometry)<br/>INDEX: spatial"]
            
            TEMP["temperature_readings<br/>---<br/>reading_id (PK)<br/>station_id (FK)<br/>observation_time<br/>temperature_c"]
        end
        
        subgraph Simulations["Simulation & Results"]
            SIMS["simulations<br/>---<br/>sim_id (PK)<br/>user_id (FK)<br/>created_at<br/>rainfall_scenario_mm<br/>model_version<br/>status<br/>exec_time_sec"]
            
            RESULTS["simulation_results<br/>---<br/>result_id (PK)<br/>sim_id (FK)<br/>computed_at<br/>avg_prob (0-1)<br/>pixels_flooded<br/>result_path"]
            
            PIXELS["flood_pixels<br/>---<br/>pixel_id (PK)<br/>result_id (FK)<br/>grid_x, grid_y<br/>flood_probability<br/>hand_value<br/>elevation<br/>built_up"]
        end
        
        subgraph Archive["Historical Archive"]
            ARCHIVE["prediction_archive<br/>---<br/>archive_id (PK)<br/>sim_id (FK)<br/>timestamp<br/>s3_location<br/>compressed_size"]
        end
    end
    
    USERS --> SESSIONS
    USERS --> SIMS
    SIMS --> RESULTS
    RESULTS --> PIXELS
    RESULTS --> ARCHIVE
    
    TOPO --> PIXELS
    PWATER --> PIXELS
    INFRA --> SIMS
    
    STATIONS --> RAINFALL
    STATIONS --> TEMP
    RAINFALL --> SIMS
    TEMP --> SIMS
    
    style PostGIS fill:#e8f5e9
    style UserMgmt fill:#c8e6c9
    style SpatialData fill:#a5d6a7
    style SensorData fill:#fff9c4
    style Simulations fill:#ffe0b2
    style Archive fill:#bbdefb
```

---

## 3.6.2 Wireframes (Dashboard Layout)

```mermaid
graph TB
    subgraph Screen["Desktop Dashboard Screen<br/>(1920×1080)"]
        subgraph Header["Header Bar (40px)"]
            LOGO["🌍 NAIROBI FLOOD TWIN"]
            USER_INFO["User: Emmanuel | Role: Responder"]
            LOGOUT["Logout"]
        end
        
        subgraph LeftPanel["Left Control Panel<br/>(300px)"]
            TITLE["⚙️ Simulation Parameters"]
            
            RAIN_LABEL["Rainfall Scenario (mm/7-day)"]
            SLIDER["[====🟢====] 50mm"]
            RAIN_RANGE["Min: 0 | Max: 200"]
            
            REGION["Select Region"]
            DROPDOWN["▼ Eastern Nairobi"]
            
            BUTTONS["━━━━━━━━━━━<br/>RUN PREDICTION<br/>━━━━━━━━━━━"]
            
            HISTORY["Recent Simulations<br/>━━━━━━━━━<br/>2026-08-24 14:32<br/>2026-08-24 13:15<br/>2026-08-23 09:42"]
        end
        
        subgraph MainMap["Main Viewport<br/>(1450×800)"]
            MAPLABEL["3D Interactive Map<br/>(WebGL + Pydeck)"]
            MAP["
            ╔════════════════════════════╗
            ║  NAIROBI FLOOD RISK MAP     ║
            ║  ═════════════════════════  ║
            ║   🔴 HIGH RISK (>0.8)       ║
            ║   🟠 MEDIUM RISK (0.5-0.8)  ║
            ║   🟡 LOW RISK (0.2-0.5)     ║
            ║   🟢 MINIMAL (<0.2)         ║
            ║                             ║
            ║  [Zoom: 100%]  [Rotate]     ║
            ║  [Reset View]  [Export]     ║
            ╚════════════════════════════╝
            "]
        end
        
        subgraph RightPanel["Right Info Panel<br/>(250px)"]
            SUMMARY_TITLE["📊 Hazard Summary"]
            STATS["
            ═══════════════════
            Affected Population:
            ~45,000 residents
            
            Critical Infrastructure:
            • 3 hospitals
            • 12 schools
            • 2 water treatment
            
            Max Flood Probability:
            0.87 (87%)
            
            Primary Flood Zone:
            Eastern Nairobi
            
            Recommended Action:
            ⚠️ EVACUATE
            ═══════════════════
            "]
            ACTIONS["[Print Report]<br/>[Download GeoJSON]<br/>[Share Team]"]
        end
        
        subgraph Footer["Footer Status Bar"]
            STATUS["✅ Model Status: Ready | Last Update: 2026-08-24 14:35:00 UTC | Compute Time: 2.34s"]
        end
    end
    
    Header --> LeftPanel
    Header --> MainMap
    Header --> RightPanel
    MainMap --> Footer
    
    style Screen fill:#f5f5f5
    style Header fill:#1a237e,color:#fff
    style LeftPanel fill:#e3f2fd
    style MainMap fill:#fff9c4
    style RightPanel fill:#f3e5f5
    style Footer fill:#37474f,color:#fff
```

---

## 3.6.3 System Architecture

```mermaid
graph TB
    subgraph Client["📱 CLIENT LAYER<br/>(Browser)"]
        BROWSER["Web Browser<br/>(Chrome/Firefox)"]
        FRONTEND["Plotly Dash<br/>Frontend Application"]
        WEBGL["WebGL + Pydeck<br/>3D Visualization"]
        INTERACT["Interactive Controls<br/>Sliders/Dropdowns"]
    end
    
    subgraph Network["🌐 NETWORK LAYER<br/>(HTTPS/REST)"]
        API_CALLS["REST API Calls<br/>POST /predict<br/>GET /hazard_summary<br/>GET /user_simulations"]
    end
    
    subgraph Backend["🖥️ BACKEND LAYER<br/>(Server)"]
        Flask["Flask API Server<br/>Port 5000"]
        
        AUTH["Authentication<br/>JWT Tokens<br/>Session Manager"]
        
        CACHE["In-Memory Cache<br/>Redis<br/>Recent Predictions"]
    end
    
    subgraph ML["🧠 ML INFERENCE LAYER"]
        PREDICTOR["Flood Predictor<br/>Pipeline"]
        MODEL["U-Net Model<br/>pytorch<br/>segmentation_model.pth"]
        PREPROCESS["Preprocessing<br/>Normalize SAR<br/>Stack Terrain"]
        POSTPROCESS["Postprocessing<br/>Threshold Probability<br/>Smooth Edges"]
    end
    
    subgraph Data["💾 DATA LAYER"]
        subgraph Spatial["Spatial Database<br/>(PostGIS)"]
            TOPO_DB["Topographic Grid<br/>HAND/Elevation/etc"]
            RESULTS_DB["Simulation Results<br/>Cache"]
        end
        
        subgraph Real["Real-Time Data"]
            RAINFALL_FEED["CHIRPS Rainfall<br/>API Feed"]
            SENSOR_STREAM["IoT Sensors<br/>Kafka Stream"]
        end
        
        subgraph Archives["Object Storage"]
            S3["AWS S3<br/>Result Archives<br/>Model Checkpoints"]
        end
    end
    
    %% Main flows
    BROWSER --> FRONTEND
    FRONTEND --> WEBGL
    FRONTEND --> INTERACT
    
    INTERACT -->|HTTP Request| API_CALLS
    API_CALLS -->|HTTPS| Flask
    
    Flask --> AUTH
    Flask --> CACHE
    Flask --> PREDICTOR
    
    PREDICTOR --> MODEL
    PREDICTOR --> PREPROCESS
    PREPROCESS --> MODEL
    MODEL --> POSTPROCESS
    
    PREPROCESS --> Spatial
    PREDICTOR --> Spatial
    POSTPROCESS --> RESULTS_DB
    
    PREDICTOR --> Real
    RAINFALL_FEED --> PREDICTOR
    SENSOR_STREAM --> PREDICTOR
    
    RESULTS_DB --> S3
    MODEL --> S3
    
    %% Return path
    POSTPROCESS -->|Response JSON| Flask
    Flask -->|HTTP Response| API_CALLS
    API_CALLS --> FRONTEND
    FRONTEND --> WEBGL
    WEBGL -->|Render| BROWSER
    
    style Client fill:#e3f2fd
    style Network fill:#fff9c4
    style Backend fill:#f3e5f5
    style ML fill:#ffe0b2
    style Data fill:#c8e6c9
    style Spatial fill:#a5d6a7
    style Real fill:#ffccbc
    style Archives fill:#b3e5fc
```

---

## Diagram References

| Section | Diagram | Purpose |
|---------|---------|---------|
| 3.5.1 | Use Case | Identifies actors and core system functions |
| 3.5.2 | Sequence | Maps information flow during prediction request |
| 3.5.3 | ERD | Database structure for data persistence |
| 3.5.4 | Class | OOP architecture for backend modules |
| 3.5.5 | Activity | User workflow for emergency response |
| 3.6.1 | Database Schema | PostGIS spatial database design |
| 3.6.2 | Wireframes | Dashboard UI layout and components |
| 3.6.3 | System Architecture | End-to-end technology pipeline |

---

## Integration with Your Implementation

All diagrams reflect your **actual system design**:

✅ **U-Net Segmentation Model** (Chapter 3.6.3 ML Layer)
- Binary flood probability prediction
- Input: 14 channels (4 SAR + 7 terrain)
- Output: Flood probability 0-1

✅ **Rainfall-Based Ground Truth** (Chapter 3.5.2 Sequence & 3.6.1 Schema)
- 7-day rainfall threshold >= 30mm
- CHIRPS data feed
- Stored in PostGIS

✅ **Interactive Dashboard** (Chapter 3.6.2 Wireframes & 3.6.3)
- Plotly Dash frontend
- WebGL 3D visualization (Pydeck)
- Real-time slider controls

✅ **REST API Backend** (Chapter 3.6.3 Backend Layer)
- Flask server
- /predict endpoint
- /hazard_summary endpoint
- JWT authentication

✅ **PostGIS Database** (Chapter 3.6.1 Schema)
- Spatial queries on topographic grid
- User simulation history
- Result caching

---

## Export Instructions

1. **Copy each diagram** into your thesis document
2. **Render at:** https://mermaid.live/
3. **Export as PNG** and insert into Chapter 3
4. **Sizing recommendations:**
   - Use Cases: 60% width
   - Sequence: 80% width (wide)
   - ERD: 70% width
   - Class Diagram: 70% width
   - Activity: 60% width
   - Database Schema: 80% width
   - Wireframes: 90% width
   - System Architecture: 90% width
