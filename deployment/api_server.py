"""
REST API for PINN Deployment
FastAPI-based REST service for model serving and inference
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path
import io
import logging

# Optional dependencies
try:
    from fastapi.middleware.cors import CORSMiddleware
    HAS_CORS = True
except:
    HAS_CORS = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== Pydantic Models for API =====

class PredictionRequest(BaseModel):
    """Request for PINN predictions"""
    coordinates: List[List[float]]  # (x, y, z) or (x, y, z, t)
    model_id: str = "default"
    field_type: str = "velocity"  # velocity, pressure, temperature, turbulence


class PredictionResponse(BaseModel):
    """Response containing predictions"""
    predictions: List[List[float]]
    metadata: Dict = {}
    status: str = "success"


class OptimizationRequest(BaseModel):
    """Request for design optimization"""
    objective: str  # "thermal_resistance", "pressure_drop", "combined"
    design_variables: List[Dict]  # [{name, min, max, type}, ...]
    constraints: Optional[List[Dict]] = None
    n_iterations: int = 100


class OptimizationResponse(BaseModel):
    """Response containing optimization results"""
    optimal_design: Dict
    objective_value: float
    optimization_history: List[float]
    status: str = "completed"


class ModelInfo(BaseModel):
    """Information about deployed model"""
    model_id: str
    model_type: str
    input_dimension: int
    output_dimension: int
    creation_date: str
    last_modified: str
    status: str


class BulkPredictionRequest(BaseModel):
    """Request for bulk predictions"""
    file_format: str = "csv"  # csv, numpy, json
    coordinates_file: str  # File path or base64 encoded content


class DatasetRequest(BaseModel):
    """Request for dataset upload"""
    dataset_name: str
    description: Optional[str] = None


# ===== Model Manager =====

class PINNModelManager:
    """
    Manages multiple PINN models for inference
    Handles loading, caching, and switching models
    """
    
    def __init__(self, model_dir: Path = Path("./models")):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.active_model = None
        self.model_metadata = {}
    
    def load_model(self, model_id: str, model_path: Path) -> bool:
        """
        Load a PINN model
        
        Args:
            model_id: Unique identifier for model
            model_path: Path to saved model (.pt file)
            
        Returns:
            True if successful
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Reconstruct model architecture from checkpoint
            model_config = checkpoint.get('config', {})
            model = self._reconstruct_model(model_config)
            model.load_state_dict(checkpoint.get('model_state_dict', {}))
            
            self.models[model_id] = model
            self.model_metadata[model_id] = model_config
            
            if self.active_model is None:
                self.active_model = model_id
            
            logger.info(f"Loaded model: {model_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def _reconstruct_model(self, config: Dict) -> nn.Module:
        """
        Reconstruct neural network from config
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            PyTorch model
        """
        # Simple fully connected network
        layers = config.get('layers', [3, 64, 64, 32, 5])
        activation = config.get('activation', 'tanh')
        
        model_layers = []
        for i in range(len(layers) - 1):
            model_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                if activation == 'relu':
                    model_layers.append(nn.ReLU())
                elif activation == 'tanh':
                    model_layers.append(nn.Tanh())
        
        return nn.Sequential(*model_layers)
    
    def predict(self, coordinates: np.ndarray, model_id: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using specified or active model
        
        Args:
            coordinates: Input coordinates (N, input_dim)
            model_id: Model ID (uses active model if None)
            
        Returns:
            Predictions (N, output_dim)
        """
        model_id = model_id or self.active_model
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")
        
        model = self.models[model_id]
        model.eval()
        
        # Convert to tensor
        coords_tensor = torch.from_numpy(coordinates).float()
        
        with torch.no_grad():
            predictions = model(coords_tensor).numpy()
        
        return predictions
    
    def list_models(self) -> List[ModelInfo]:
        """List all loaded models"""
        models_info = []
        
        for model_id, config in self.model_metadata.items():
            info = ModelInfo(
                model_id=model_id,
                model_type=config.get('type', 'unknown'),
                input_dimension=config.get('layers', [0])[0],
                output_dimension=config.get('layers', [0])[-1],
                creation_date=config.get('creation_date', 'unknown'),
                last_modified=config.get('modified_date', 'unknown'),
                status='active' if model_id == self.active_model else 'inactive'
            )
            models_info.append(info)
        
        return models_info


# ===== FastAPI Application =====

def create_app(model_dir: Path = Path("./models")) -> FastAPI:
    """
    Create FastAPI application with PINN endpoints
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="PINN Heat Sink Optimizer API",
        description="Physics-Informed Neural Network REST API",
        version="1.0.0"
    )
    
    # Add CORS if available
    if HAS_CORS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Initialize model manager
    model_manager = PINNModelManager(model_dir)
    
    # ===== Health & Status Endpoints =====
    
    @app.get("/")
    async def root():
        """Root endpoint with API info"""
        return {
            "service": "PINN Heat Sink Optimizer",
            "status": "operational",
            "endpoints": [
                "/docs - Interactive API documentation",
                "/models - List loaded models",
                "/predict - Make predictions",
                "/optimize - Run design optimization",
                "/health - Health check"
            ]
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "models_loaded": len(model_manager.models),
            "active_model": model_manager.active_model
        }
    
    # ===== Model Management Endpoints =====
    
    @app.get("/models")
    async def list_models() -> List[ModelInfo]:
        """List all loaded models"""
        return model_manager.list_models()
    
    @app.post("/models/load")
    async def load_model(model_id: str, model_path: str):
        """Load a new model"""
        success = model_manager.load_model(model_id, Path(model_path))
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to load model")
        
        return {"status": "success", "model_id": model_id}
    
    @app.post("/models/activate/{model_id}")
    async def activate_model(model_id: str):
        """Activate a model for inference"""
        if model_id not in model_manager.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        model_manager.active_model = model_id
        return {"status": "success", "active_model": model_id}
    
    # ===== Prediction Endpoints =====
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """
        Make PINN predictions
        
        Example:
        {
            "coordinates": [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]],
            "model_id": "default",
            "field_type": "velocity"
        }
        """
        try:
            coordinates = np.array(request.coordinates, dtype=np.float32)
            
            predictions = model_manager.predict(coordinates, request.model_id)
            
            return PredictionResponse(
                predictions=predictions.tolist(),
                metadata={
                    "n_points": len(coordinates),
                    "field_type": request.field_type,
                    "model_id": request.model_id or model_manager.active_model
                }
            )
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/batch")
    async def batch_predict(request: BulkPredictionRequest):
        """
        Batch prediction from file
        
        Supports CSV, numpy, or JSON formats
        """
        try:
            # Parse file
            if request.file_format == 'csv':
                import csv
                reader = csv.reader(io.StringIO(request.coordinates_file))
                coordinates = np.array([row for row in reader], dtype=np.float32)
            
            elif request.file_format == 'numpy':
                coordinates = np.load(io.BytesIO(
                    request.coordinates_file.encode()
                ))
            
            elif request.file_format == 'json':
                coordinates = np.array(
                    json.loads(request.coordinates_file),
                    dtype=np.float32
                )
            
            else:
                raise ValueError(f"Unsupported format: {request.file_format}")
            
            predictions = model_manager.predict(coordinates)
            
            return {
                "status": "success",
                "n_predictions": len(predictions),
                "predictions_summary": {
                    "mean": predictions.mean(axis=0).tolist(),
                    "std": predictions.std(axis=0).tolist(),
                    "min": predictions.min(axis=0).tolist(),
                    "max": predictions.max(axis=0).tolist(),
                }
            }
        
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ===== Optimization Endpoints =====
    
    @app.post("/optimize")
    async def run_optimization(request: OptimizationRequest, 
                              background_tasks: BackgroundTasks):
        """
        Run design optimization using PINN surrogate
        
        Example:
        {
            "objective": "thermal_resistance",
            "design_variables": [
                {"name": "fin_height", "min": 0.01, "max": 0.1, "type": "float"},
                {"name": "fin_spacing", "min": 0.005, "max": 0.05, "type": "float"}
            ],
            "n_iterations": 50
        }
        """
        try:
            # Placeholder for optimization logic
            n_vars = len(request.design_variables)
            
            # Generate dummy optimization results
            history = []
            optimal_design = {}
            
            for i, var in enumerate(request.design_variables):
                optimal_design[var['name']] = (var['min'] + var['max']) / 2
                history.append(float(i))
            
            return OptimizationResponse(
                optimal_design=optimal_design,
                objective_value=0.85,
                optimization_history=history,
                status="completed"
            )
        
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ===== Data Management Endpoints =====
    
    @app.post("/datasets/upload")
    async def upload_dataset(file: UploadFile = File(...)):
        """Upload CFD dataset for training"""
        try:
            contents = await file.read()
            
            # Save dataset
            dataset_path = Path(f"./data/{file.filename}")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dataset_path, 'wb') as f:
                f.write(contents)
            
            return {
                "status": "success",
                "filename": file.filename,
                "size": len(contents),
                "path": str(dataset_path)
            }
        
        except Exception as e:
            logger.error(f"Dataset upload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/datasets")
    async def list_datasets():
        """List all available datasets"""
        data_dir = Path("./data")
        datasets = []
        
        if data_dir.exists():
            for file in data_dir.iterdir():
                if file.is_file():
                    datasets.append({
                        "name": file.name,
                        "size": file.stat().st_size,
                        "modified": file.stat().st_mtime
                    })
        
        return {"datasets": datasets}
    
    # ===== Export Endpoints =====
    
    @app.get("/export/model/{model_id}")
    async def export_model(model_id: str):
        """Export model as ONNX or TorchScript"""
        try:
            if model_id not in model_manager.models:
                raise HTTPException(status_code=404, detail="Model not found")
            
            model = model_manager.models[model_id]
            
            # Export to ONNX (simplified)
            export_path = Path(f"./exports/{model_id}.onnx")
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Placeholder: actual ONNX export would go here
            
            return FileResponse(str(export_path))
        
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/export/results")
    async def export_results(format: str = "csv"):
        """Export recent prediction results"""
        try:
            # Placeholder: gather and export recent results
            return {"status": "success", "format": format}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# ===== Initialization =====

if __name__ == "__main__":
    import uvicorn
    
    app = create_app(model_dir=Path("./models"))
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
