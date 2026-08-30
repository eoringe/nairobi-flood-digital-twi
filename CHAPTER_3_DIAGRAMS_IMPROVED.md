# Chapter 3: Analysis & Design Diagrams (IMPROVED LAYOUTS)
## Nairobi Urban Flood Digital Twin

---

# 3.5 ANALYSIS DIAGRAMS (IMPROVED)

## 3.5.1 Use Case Diagram (Cleaner Layout — No Line Crossing)

```mermaid
graph LR
    subgraph Actors["👥 ACTORS"]
        Citizen["👤 Citizen"]
        Planner["👔 Planner"]
        Responder["🚨 Responder"]
        Admin["⚙️ Admin"]
    end
    
    subgraph System["🗺️ NAIROBI FLOOD DIGITAL TWIN"]
        UC1["View Real-time<br/>Flood Risk Map"]
        UC2["View Flood<br/>Forecast"]
        UC3["Adjust Rainfall<br/>Scenario Sliders"]
        UC4["View 3D Flood<br/>Projections"]
        UC5["Retrieve Localized<br/>Hazard Summaries"]
        UC6["Generate<br/>Emergency Reports"]
        UC7["Manage User<br/>Accounts"]
        UC8["Update Sensor<br/>Data Streams"]
    end
    
    Citizen -->|views| UC1
    Citizen -->|views| UC2
    
    Planner -->|uses| UC3
    Planner -->|uses| UC4
    Planner -->|uses| UC6
    
    Responder -->|uses| UC1
    Responder -->|uses| UC4
    Responder -->|uses| UC5
    Responder -->|uses| UC6
    
    Admin -->|manages| UC7
    Admin -->|manages| UC8
    
    style Actors fill:#fff3e0
    style System fill:#e3f2fd
    style Citizen fill:#e8f5e9
    style Planner fill:#fff3e0
    style Responder fill:#ffebee
    style Admin fill:#f3e5f5
```

---

## 3.5.2 Sequence Diagram (Same — Already Clean)

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

## 3.5.3 Entity-Relationship Diagram (Horizontal Compact Layout — Fits Half A4)

```mermaid
graph LR
    subgraph Users["👥 USER & AUTH"]
        U["<b>users</b><br/>user_id (PK)<br/>username<br/>email<br/>role"]
        S["<b>user_sessions</b><br/>session_id (PK)<br/>user_id (FK)<br/>token<br/>expires_at"]
    end
    
    subgraph DataSources["📡 DATA SOURCES"]
        SENSOR["<b>sensor_data</b><br/>sensor_id (PK)<br/>sensor_name<br/>latitude, longitude<br/>status"]
        
        RAIN_FEED["<b>meteorological_feeds</b><br/>feed_id (PK)<br/>sensor_id (FK)<br/>value (rainfall_mm)<br/>observation_time"]
    end
    
    subgraph Processing["⚙️ SIMULATIONS"]
        SIM["<b>simulations</b><br/>sim_id (PK)<br/>user_id (FK)<br/>feed_id (FK)<br/>rainfall_scenario<br/>model_version<br/>status"]
        
        RES["<b>results</b><br/>result_id (PK)<br/>sim_id (FK)<br/>computed_at<br/>avg_flood_prob<br/>result_path"]
    end
    
    subgraph SpatialData["🗺️ GEOSPATIAL LAYERS"]
        TOPO["<b>topographic_grid</b><br/>cell_id (PK)<br/>grid_x, grid_y<br/>elevation<br/>hand<br/>slope<br/>built_up<br/>geom"]
        
        PWATER["<b>permanent_water</b><br/>water_id (PK)<br/>area<br/>geom"]
        
        PIXELS["<b>flood_pixels</b><br/>pixel_id (PK)<br/>result_id (FK)<br/>grid_x, grid_y<br/>flood_probability<br/>elevation<br/>hand_value"]
    end
    
    subgraph Archive["📦 ARCHIVE"]
        ARCH["<b>prediction_archive</b><br/>archive_id (PK)<br/>sim_id (FK)<br/>s3_location<br/>timestamp"]
    end
    
    %% User relationships
    U --> S
    U --> SIM
    
    %% Data source relationships
    SENSOR --> RAIN_FEED
    RAIN_FEED --> SIM
    
    %% Simulation flow
    SIM --> RES
    
    %% Results to spatial
    RES --> PIXELS
    RES --> ARCH
    
    %% Spatial relationships
    TOPO --> PIXELS
    PWATER --> PIXELS
    
    %% Styling
    style Users fill:#e8f5e9
    style DataSources fill:#fff9c4
    style Processing fill:#ffe0b2
    style SpatialData fill:#bbdefb
    style Archive fill:#f3e5f5
    style U fill:#c8e6c9
    style S fill:#c8e6c9
    style SENSOR fill:#ffd54f
    style RAIN_FEED fill:#ffd54f
    style SIM fill:#ffb74d
    style RES fill:#ffb74d
    style TOPO fill:#64b5f6
    style PWATER fill:#64b5f6
    style PIXELS fill:#64b5f6
    style ARCH fill:#ce93d8
```

**Entity Summary:**
| Entity | Purpose | Key Fields |
|--------|---------|-----------|
| **users** | User accounts & authentication | user_id, username, email, role |
| **user_sessions** | Active login sessions | session_id, token, expires_at |
| **sensor_data** | IoT/weather stations metadata | sensor_id, location, status |
| **meteorological_feeds** | Real-time rainfall/temperature | feed_id, value, observation_time |
| **simulations** | Flood prediction runs | sim_id, rainfall_scenario, status |
| **results** | Prediction output | result_id, avg_flood_prob, result_path |
| **topographic_grid** | Terrain data (HAND/elevation/slope) | cell_id, grid_x/y, elevation, hand |
| **permanent_water** | Static water bodies | water_id, area, geom |
| **flood_pixels** | Per-pixel predictions | pixel_id, flood_probability, elevation |
| **prediction_archive** | Historical storage | archive_id, s3_location, timestamp |

---

## 3.5.4 Class Diagram (Redesigned — Minimal Line Crossing)

```mermaid
classDiagram
    %% ========== FRONTEND LAYER ==========
    class DashboardApp {
        -app: dash.Dash
        -layout: html.Div
        -callbacks: dict
        +register_callback(): void
        +run_server(): void
        +update_map(): void
    }
    
    class GeospatialRenderer {
        -canvas: WebGLContext
        -data_layer: GeoJSON
        -color_scale: ColorMap
        +render_flood_map(): void
        +update_overlay(): void
        +add_marker(): void
        +zoom_to_bounds(): void
    }
    
    %% ========== API & AUTH LAYER ==========
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
    
    %% ========== PREDICTION PIPELINE ==========
    class FloodPredictor {
        -model: SurrogateModel
        -telemetry: TelemetryData
        -db: DatabaseManager
        +predict(): np.ndarray
        +postprocess_results(): dict
        +validate_spatial_plausibility(): bool
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
    
    class DatabaseManager {
        -connection: psycopg2.connection
        -postgis_enabled: bool
        +query_topography(): dict
        +store_result(): void
        +get_user_simulations(): list
        +spatial_query(): GeoDataFrame
    }
    
    %% ========== RELATIONSHIPS (Minimal Crossings) ==========
    %% Frontend to API
    DashboardApp --> GeospatialRenderer
    DashboardApp --> APIGateway
    
    %% API to Auth
    APIGateway --> AuthManager
    
    %% API to Predictor
    APIGateway --> FloodPredictor
    
    %% Predictor to Components (Vertical stack = no crossing)
    FloodPredictor --> SurrogateModel
    FloodPredictor --> TelemetryData
    FloodPredictor --> DatabaseManager
    
    %% Data sharing
    DatabaseManager --> TelemetryData
    GeospatialRenderer --> DatabaseManager
```

---

## 3.5.5 Activity Diagram (Horizontal Flow — Left to Right)

```mermaid
graph LR
    START([Login]) --> AUTH{Credentials<br/>Valid?}
    
    AUTH -->|No| INVALID["❌ Invalid<br/>Show Error"]
    INVALID --> RETRY["Retry Login"]
    RETRY --> AUTH
    
    AUTH -->|Yes| DASHBOARD["Load Dashboard<br/>& Map"]
    
    DASHBOARD --> SELECT["Select<br/>Catchment"]
    
    SELECT --> PARAM["View Sensor<br/>Data"]
    
    PARAM --> ADJUST["Adjust Rainfall<br/>Scenario<br/>0-200mm"]
    
    ADJUST --> REQUEST["Submit<br/>Rainfall Input"]
    
    REQUEST --> PREDICT["Run U-Net Model<br/>~2 sec"]
    
    PREDICT --> RENDER["Render 3D<br/>WebGL Map"]
    
    RENDER --> VIEW_RESULTS["Display Results<br/>& Grid"]
    
    VIEW_RESULTS --> HAZARD["Generate<br/>Summary Report"]
    
    HAZARD --> QUERY["Query Specific<br/>Zones"]
    
    QUERY --> STATS{Need More<br/>Info?}
    
    STATS -->|Yes| QUERY
    
    STATS -->|No| EXPORT["Export<br/>GeoJSON"]
    
    EXPORT --> END_SUCCESS["✅ Complete"]
    
    END_SUCCESS --> END_NEW["New Scenario<br/>or Exit"]
    
    style START fill:#c8e6c9
    style AUTH fill:#fff9c4
    style DASHBOARD fill:#bbdefb
    style PREDICT fill:#ffe0b2
    style RENDER fill:#f8bbd0
    style END_SUCCESS fill:#c8e6c9
    style INVALID fill:#ffcdd2
    style END_NEW fill:#e0e0e0
```

---

# 3.6 DESIGN DIAGRAMS (SAME)

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

## 📊 Comparison: Original vs Improved

| Diagram | Improvement |
|---------|-------------|
| **Use Case 3.5.1** | ✅ Changed to LR (left-right) layout with actors on left, system in middle. Eliminated all line crossings. |
| **Activity 3.5.5** | ✅ Changed from TD (top-down) to LR (left-right horizontal flow). Much easier to read linearly. |
| **Class 3.5.4** | ✅ Reorganized classes into logical layers (Frontend, API, Prediction, Database). Relationships only connect within/between layers, no crossing lines. |
| **Sequence 3.5.2** | ✓ Already optimal (chronological left-to-right) |
| **ERD 3.5.3** | ✓ Already optimal (entity-relationship standard) |
| **Database 3.6.1** | ✓ Already optimal (logical grouping by table type) |
| **Wireframes 3.6.2** | ✓ Already optimal (spatial layout) |
| **Architecture 3.6.3** | ✓ Already optimal (layered stack) |

---

## Export Instructions

1. **Copy improved diagram code** into Mermaid Live: https://mermaid.live/
2. **Preview to verify clarity**
3. **Export as PNG** and insert into Chapter 3
4. **Recommended sizes:**
   - Use Cases (LR): 70% width
   - Activity (LR): 90% width (horizontal flow)
   - Class Diagram: 75% width (no crossing)
   - Others: as per original guide
