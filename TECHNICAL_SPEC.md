# 📋 NFL Big Data Bowl 2026 - Technical Specification Document

## 🎯 Competition Requirements (From Kaggle Analysis)

### Core Challenge
**Predict player (x, y) coordinates for each frame while ball is in the air**
- Input: Pre-pass tracking data (before QB release)  
- Context: Targeted receiver + ball landing location provided
- Output: Player positions for 0.5-4 second trajectory sequences
- Evaluation: RMSE on coordinate predictions
- Timeline: Live evaluation on Dec 4 - Jan 5 NFL games

### Data Specifications

#### Input Schema
```python
# input_2023_w[01-18].csv - Pre-pass tracking
class InputSchema:
    # Core identifiers
    game_id: int                    # Unique game identifier
    play_id: int                    # Play identifier (not unique across games)
    nfl_id: int                     # Unique player ID
    frame_id: int                   # Frame sequence number (10 FPS)
    
    # Prediction metadata
    player_to_predict: bool         # Whether this player needs prediction
    num_frames_output: int          # Number of frames to predict (5-40 typical)
    
    # Spatial coordinates (primary features)
    x: float                        # Long axis position [0-120 yards]
    y: float                        # Short axis position [0-53.3 yards]
    
    # Motion vectors
    s: float                        # Speed [yards/second]
    a: float                        # Acceleration [yards/second²]
    o: float                        # Player orientation [degrees]
    dir: float                      # Direction of motion [degrees]
    
    # Game context
    play_direction: str             # Offense moving 'left' or 'right'
    absolute_yardline_number: int   # Distance from endzone [0-100]
    
    # Player attributes
    player_height: str              # Format: "ft-in" (e.g., "6-2")
    player_weight: int              # Pounds
    player_birth_date: str          # Format: "yyyy-mm-dd"
    player_position: str            # Position code (QB, RB, WR, etc.)
    player_side: str                # "Offense" or "Defense"
    player_role: str                # Play-specific role
    
    # Target information
    ball_land_x: float              # Ball landing x-coordinate
    ball_land_y: float              # Ball landing y-coordinate
```

#### Output Schema (Training Labels)
```python
# output_2023_w[01-18].csv - Post-pass tracking (TARGETS)
class OutputSchema:
    game_id: int                    # Links to input data
    play_id: int                    # Links to input data  
    nfl_id: int                     # Links to input data
    frame_id: int                   # Sequential frame [1 to num_frames_output]
    
    # PREDICTION TARGETS
    x: float                        # Player x-position to predict
    y: float                        # Player y-position to predict
```

#### Test Schema
```python
# test_input.csv - Evaluation data (no ground truth)
# Same structure as input files

# test.csv - Prediction template
class TestSchema:
    game_id: int
    play_id: int
    nfl_id: int
    frame_id: int
    # Note: x, y need to be predicted

# sample_submission.csv - Submission format
class SubmissionSchema:
    id: str                         # "{game_id}_{play_id}_{nfl_id}_{frame_id}"
    x: float                        # Predicted x-coordinate
    y: float                        # Predicted y-coordinate
```

## 🏗️ Technical Architecture Specifications

### 1. Data Processing Pipeline

#### A. Raw Data Validation
```python
class DataValidator:
    """Validate tracking data quality and consistency"""
    
    def validate_spatial_bounds(self, df):
        # Field dimensions: 120 x 53.3 yards
        assert df['x'].between(0, 120).all()
        assert df['y'].between(0, 53.3).all()
    
    def validate_temporal_consistency(self, df):
        # 10 FPS = 0.1 second intervals
        # Check frame sequence completeness
        pass
    
    def validate_physics_constraints(self, df):
        # Speed limits: < 12 yards/second (24 mph)
        # Acceleration limits: < 20 yards/second²
        pass
```

#### B. Feature Engineering Pipeline
```python
class FeatureEngineer:
    """Extract spatiotemporal features for trajectory prediction"""
    
    def extract_velocity_features(self, df):
        # Velocity vectors, acceleration, jerk
        # Smoothed derivatives using Savitzky-Golay filters
        pass
    
    def extract_relative_positions(self, df):
        # Distance to ball landing location
        # Relative positions to other players
        # Angles and bearings
        pass
        
    def extract_contextual_features(self, df):  
        # Time to ball arrival
        # Field zone encoding
        # Formation analysis
        pass
        
    def extract_graph_features(self, df):
        # Player interaction networks
        # Nearest neighbor analysis  
        # Coverage assignments
        pass
```

### 2. Model Architecture Specifications

#### A. Physics-Informed Neural Network
```python
class TrajectoryPINN(nn.Module):
    """Physics-constrained trajectory prediction"""
    
    def __init__(self, input_dim=32, hidden_dim=256, sequence_length=20):
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 2)  # (x, y) coordinates
        
        # Physics constraint layers
        self.velocity_constraint = VelocityConstraintLayer()
        self.acceleration_constraint = AccelerationConstraintLayer()
        self.boundary_constraint = FieldBoundaryLayer()
    
    def forward(self, x):
        # Standard sequence prediction
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        positions = self.output_layer(decoded)
        
        # Apply physics constraints
        positions = self.velocity_constraint(positions)
        positions = self.acceleration_constraint(positions) 
        positions = self.boundary_constraint(positions)
        
        return positions

class VelocityConstraintLayer(nn.Module):
    """Enforce realistic velocity limits"""
    def forward(self, positions):
        # Max speed: 12 yards/second
        # Smooth velocity transitions
        pass

class AccelerationConstraintLayer(nn.Module):  
    """Enforce realistic acceleration limits"""
    def forward(self, positions):
        # Max acceleration: 20 yards/second²
        # Physics-based motion equations
        pass
```

#### B. Graph Attention Network
```python
class PlayerInteractionGAT(nn.Module):
    """Model player interactions using graph attention"""
    
    def __init__(self, node_features=16, edge_features=8, num_heads=8):
        self.node_embedding = nn.Linear(node_features, 64)
        self.edge_embedding = nn.Linear(edge_features, 64)
        
        self.gat_layers = nn.ModuleList([
            GATConv(64, 64, heads=num_heads, concat=False)
            for _ in range(3)
        ])
        
        self.output_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # (x, y) prediction
        )
    
    def build_player_graph(self, player_positions, max_distance=10):
        """Build dynamic graph based on player proximities"""
        # Create edges between players within max_distance
        # Edge features: relative distance, angle, velocity difference
        pass
    
    def forward(self, player_data):
        # Build graph per timestep
        graph = self.build_player_graph(player_data)
        
        # Node embeddings
        node_emb = self.node_embedding(graph.x)
        
        # Graph attention layers
        for gat_layer in self.gat_layers:
            node_emb = gat_layer(node_emb, graph.edge_index)
            
        # Predict trajectories
        return self.output_mlp(node_emb)
```

#### C. Transformer-XL Temporal Model
```python
class TrajectoryTransformerXL(nn.Module):
    """Extended transformer for long-range temporal dependencies"""
    
    def __init__(self, d_model=512, n_head=8, n_layers=6, seq_len=50):
        self.position_encoding = SpatialTemporalEncoding(d_model)
        self.transformer = TransformerXL(
            d_model=d_model, 
            n_head=n_head,
            n_layer=n_layers,
            mem_len=seq_len//2  # Extended memory for long sequences
        )
        self.output_projection = nn.Linear(d_model, 2)
    
    def forward(self, x, memory=None):
        # Add spatial-temporal position encoding
        x = self.position_encoding(x)
        
        # Extended transformer with memory
        output, new_memory = self.transformer(x, memory)
        
        # Project to coordinates
        trajectory = self.output_projection(output)
        
        return trajectory, new_memory

class SpatialTemporalEncoding(nn.Module):
    """Position encoding for field coordinates and time"""
    def __init__(self, d_model):
        # Combine spatial (x,y field position) and temporal encodings
        pass
```

### 3. Ensemble Architecture

#### A. Multi-Model Ensemble
```python
class TrajectoryEnsemble(nn.Module):
    """Weighted ensemble of specialized models"""
    
    def __init__(self):
        self.pinn_model = TrajectoryPINN()
        self.gat_model = PlayerInteractionGAT()
        self.transformer_model = TrajectoryTransformerXL()
        self.lstm_model = ConvLSTMTrajectory()
        
        # Meta-learning for dynamic weights
        self.meta_learner = MetaWeightNetwork(
            context_features=64,  # Play situation, player types, etc.
            num_models=4
        )
    
    def forward(self, x, context):
        # Get predictions from each model
        pinn_pred = self.pinn_model(x)
        gat_pred = self.gat_model(x) 
        transformer_pred = self.transformer_model(x)
        lstm_pred = self.lstm_model(x)
        
        # Dynamic ensemble weights based on context
        weights = self.meta_learner(context)
        
        # Weighted combination
        ensemble_pred = (
            weights[:, 0:1] * pinn_pred +
            weights[:, 1:2] * gat_pred + 
            weights[:, 2:3] * transformer_pred +
            weights[:, 3:4] * lstm_pred
        )
        
        return ensemble_pred

class MetaWeightNetwork(nn.Module):
    """Learn ensemble weights based on play context"""
    def __init__(self, context_features, num_models):
        self.context_encoder = nn.Sequential(
            nn.Linear(context_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_models),
            nn.Softmax(dim=-1)  # Normalize weights
        )
    
    def forward(self, context):
        return self.context_encoder(context)
```

### 4. Training Infrastructure

#### A. Cross-Validation Strategy
```python
class TemporalGameSplit:
    """Prevent data leakage with game-level temporal splits"""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
    
    def split(self, X, y, groups):
        # Split by weeks: 1-14 train, 15-18 validation
        # Never split within same game
        # Stratify by down/distance situations
        pass

class PlayerGeneralizationSplit:
    """Test generalization to unseen players"""
    
    def split_by_player_familiarity(self, df):
        # Holdout certain players entirely for validation
        # Test model's ability to predict new player movements
        pass
```

#### B. Loss Functions
```python
class TrajectoryLoss(nn.Module):
    """Multi-component loss for trajectory prediction"""
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05):
        self.alpha = alpha  # Coordinate accuracy weight
        self.beta = beta    # Velocity consistency weight  
        self.gamma = gamma  # Physics constraint weight
    
    def forward(self, pred_traj, true_traj):
        # Primary RMSE loss
        coord_loss = F.mse_loss(pred_traj, true_traj)
        
        # Velocity consistency loss
        pred_vel = torch.diff(pred_traj, dim=1)
        true_vel = torch.diff(true_traj, dim=1)
        velocity_loss = F.mse_loss(pred_vel, true_vel)
        
        # Physics constraint loss
        physics_loss = self.compute_physics_violations(pred_traj)
        
        total_loss = (
            self.alpha * coord_loss + 
            self.beta * velocity_loss +
            self.gamma * physics_loss
        )
        
        return total_loss
    
    def compute_physics_violations(self, trajectory):
        # Penalize unrealistic speeds/accelerations
        # Encourage smooth, realistic motion
        pass
```

### 5. Inference Pipeline Specifications

#### A. Real-Time Prediction Service
```python
class LivePredictionAPI:
    """Production inference for live NFL games"""
    
    def __init__(self, model_path, batch_size=32):
        self.ensemble = self.load_ensemble(model_path)
        self.feature_pipeline = FeatureEngineer()
        self.batch_size = batch_size
    
    async def predict_trajectories(self, game_data):
        # Process input features
        features = self.feature_pipeline.transform(game_data)
        
        # Batch prediction
        predictions = []
        for batch in self.batch_generator(features):
            batch_pred = self.ensemble(batch)
            predictions.append(batch_pred)
        
        # Format for Kaggle submission
        submission = self.format_submission(predictions)
        return submission
    
    def format_submission(self, predictions):
        # Convert to required CSV format: id, x, y
        pass
```

#### B. Performance Monitoring
```python
class ModelMonitor:
    """Monitor model performance and drift"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.drift_detector = StatisticalDriftDetector()
    
    def monitor_prediction_quality(self, predictions, actuals=None):
        # Track RMSE, latency, throughput
        # Detect anomalous predictions
        pass
    
    def detect_model_drift(self, current_features, reference_features):
        # Statistical tests for feature drift
        # Alert if model needs retraining
        pass
```

## 🎯 Performance Targets

### Accuracy Targets
- **Primary Metric**: RMSE < 0.8 yards (historical top 1%)
- **Per-Player Type**: Specialized accuracy for QB, RB, WR, defensive players
- **Temporal Accuracy**: Consistent accuracy across all prediction horizons (0.5-4 seconds)

### Computational Targets  
- **Training Time**: < 24 hours on 8x A100 GPUs
- **Inference Latency**: < 50ms per player per frame
- **Memory Usage**: < 16GB GPU memory for inference
- **Throughput**: > 1000 predictions per second

### Robustness Targets
- **Cross-Validation**: < 5% variance across folds
- **Player Generalization**: < 10% accuracy drop on unseen players  
- **Temporal Generalization**: < 5% accuracy drop on future weeks
- **Situational Robustness**: Consistent performance across down/distance situations

This technical specification provides the detailed blueprint for implementing our competition-winning solution, with emphasis on accuracy, scalability, and real-world deployment considerations.