from typing import List
import numpy as np

from src.shared.model.model_data import ModelData
import src.shared.utils as utils


from src.server.aggregator.model_aggregator import ModelAggregator
from src.shared.model.model_meta import ModelMeta

class FedAvgAggregator(ModelAggregator):
    def __init__(self):
        pass

    def aggregate(self, base_model_data: ModelData, model_data_list: List[ModelData]) -> ModelData:
        super().aggregate(base_model_data, model_data_list)
        
        # Extract the weights from all models
        weights_list = [md._get_weigths_np() for md in model_data_list]
        num_samples_epochs_learned_aggregated = [md.num_samples_epochs_learned for md in model_data_list]

        # Calculate the total number of samples
        total_samples = sum(num_samples_epochs_learned_aggregated)

        print(f"Total samples: {total_samples}")

        # Initialize the aggregated weights with zeros (empty model as base)
        aggregated_weights = utils.get_model().get_weights()
        
        # Weighted sum of all the weights
        for i in range(len(aggregated_weights)):
            aw = np.zeros_like(aggregated_weights[i])
            for j in range(len(weights_list)):
                aw_c = weights_list[j][i] * (num_samples_epochs_learned_aggregated[j] / (total_samples * 1.))
                aw += aw_c
            aggregated_weights[i] = aw

        # Aggregate metadata
        aggregated_meta = self._aggregate_metadata(model_data_list)
        
        # Create a new ModelData instance with averaged weights
        aggregated_model_data = ModelData(aggregated_meta, aggregated_weights)
        
        return aggregated_model_data

    def _aggregate_metadata(self, model_data_list: List[ModelData]) -> ModelMeta:
        # Aggregate metadata
        num_samples_list = [md.num_samples_learned for md in model_data_list]
        num_epochs_list = [md.num_epochs_learned for md in model_data_list]
        num_round_list = [md.num_round for md in model_data_list]
        num_samples_epochs_learned_list = [md.num_samples_epochs_learned for md in model_data_list]
        learned_dates_list = [md.learned_dates for md in model_data_list]

        
        # Calculate the total number of samples
        total_samples = sum(num_samples_list)

        # Aggregate metadata
        aggregated_num_samples = total_samples
        aggregated_num_epochs = sum(num_epochs_list)
        aggregated_num_round = sum(num_round_list)
        num_samples_epochs_learned = sum(num_samples_epochs_learned_list)
        aggregated_learned_dates = list(set(date for sublist in learned_dates_list for date in sublist))
        
        # Create a new ModelMeta instance with the aggregated metadata
        aggregated_meta = ModelMeta(
            num_samples_learned=aggregated_num_samples,
            num_epochs_learned=aggregated_num_epochs,
            num_round=aggregated_num_round,
            num_samples_epochs_learned=num_samples_epochs_learned,
            learned_dates=aggregated_learned_dates
        )
        
        return aggregated_meta