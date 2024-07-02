from typing import List
import numpy as np

from src.shared.model.model_data import ModelData
import src.shared.utils as utils

from src.server.aggregator.model_aggregator import ModelAggregator

class FedDeltaAggregator(ModelAggregator):
    def __init__(self):
        pass

    def aggregate(self, base_model_data: ModelData, model_data_list: List[ModelData]) -> ModelData:
        super().aggregate(base_model_data, model_data_list)

        aggregated_weights = base_model_data._get_weigths_np()
        
        # Aggregate weights using FedSGD
        for model_data in model_data_list:
            client_weights = model_data._get_weigths_np()
            
            # Update the global model with client weights using SGD
            for i in range(len(aggregated_weights)):
                aggregated_weights[i] += client_weights[i]
            
            base_model_data.num_epochs_learned += model_data.num_epochs_learned
            base_model_data.num_samples_learned += model_data.num_samples_learned
            base_model_data.num_samples_epochs_learned += model_data.num_samples_epochs_learned
            base_model_data.num_round += model_data.num_round
            base_model_data.learned_dates = []

        # Create a new ModelData instance with aggregated weights
        aggregated_model_data = ModelData(base_model_data, aggregated_weights)
        
        return aggregated_model_data
