"""
Federated Learning Server for Heart Disease Prediction
Enables multi-hospital training without data sharing
"""
import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
from flwr.common import Metrics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedHeartServer:
    """Federated learning server for heart disease prediction"""
    
    def __init__(self):
        self.strategy = fl.server.strategy.FedAvg(
            min_available_clients=2,
            min_fit_clients=2,
            min_eval_clients=2,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=self.weighted_average,
            on_fit_config_fn=self.get_fit_config,
            on_evaluate_config_fn=self.get_evaluate_config,
        )
    
    def get_fit_config(self, server_round: int) -> Dict:
        """Return training configuration for each round"""
        config = {
            "batch_size": 32,
            "current_round": server_round,
            "local_epochs": 3,
            "learning_rate": 0.01,
        }
        return config
    
    def get_evaluate_config(self, server_round: int) -> Dict:
        """Return evaluation configuration for each round"""
        config = {
            "batch_size": 32,
            "eval_round": server_round,
        }
        return config
    
    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate metrics from multiple clients with weighting"""
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        
        # Aggregate and return custom metric
        return {"accuracy": sum(accuracies) / sum(examples)}
    
    def start_server(self, port: int = 8080):
        """Start the federated learning server"""
        logger.info(f"Starting Federated Learning server on port {port}")
        
        try:
            fl.server.start_server(
                server_address=f"0.0.0.0:{port}",
                config=fl.server.ServerConfig(num_rounds=10),
                strategy=self.strategy,
            )
            logger.info("Federated Learning server started successfully")
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            raise

if __name__ == "__main__":
    server = FederatedHeartServer()
    server.start_server(port=8080)