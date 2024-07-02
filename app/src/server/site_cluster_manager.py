import os
import pickle
import threading

from src.shared.model.site import Site
from src.server.aggregator.fed_avg_dual_aggregator import FedAvgDualAggregator
from src.shared.model.model_data import ModelData
from src.shared import utils
from src.shared.model.model_meta import ModelMeta


class SiteClusterManager:
    def __init__(self, sites, data_path):
        self.data_path = data_path

        self.aggregator = FedAvgDualAggregator()

        self.sites = {}
        self.site_online_states = {}
        
        self.clusters = {}
        self.cluster_models = {}

        self.global_model = None

        # Add locks for sites, clusters and global model
        self.site_locks = {}
        self.cluster_locks = {}
        self.global_model_lock = threading.Lock()

        for site in sites:
            self.sites[site.site_id] = site

            if site.cluster not in self.clusters:
                model_data_path = f"{self.data_path}/cluster_{site.cluster}_model_data.pkl"
                if os.path.exists(model_data_path):
                    with open(model_data_path) as f:
                        model_data = pickle.load(f)
                else:
                    model_data = ModelData(ModelMeta(), utils.get_model().get_weights())

                self.clusters[site.cluster] = []
                self.cluster_models[site.cluster] = model_data
                self.cluster_locks[site.cluster] = threading.Lock()

            self.clusters[site.cluster].append(site.site_id)
        
        model_data_path = f"{self.data_path}/global_model_data.pkl"
        if os.path.exists(model_data_path):
            with open(model_data_path) as f:
                model_data = pickle.load(f)
        else:
            model_data = ModelData(ModelMeta(), utils.get_model().get_weights())
        
        self.global_model = model_data

    def set_site_online(self, site_id):
        self.site_online_states[site_id] = True

    def set_site_offline(self, site_id):
        self.site_online_states[site_id] = False

    def is_site_online(self, site_id):
        return self.site_online_states.get(site_id, False)
    
    def get_sites_in_cluster(self, cluster):
        return self.clusters.get(cluster, None)
    
    def is_site_known(self, site_id):
        return site_id in self.sites
    
    def get_cluster_for_site(self, site_id):
        return self.sites[site_id].cluster
    
    def update_cluster_model(self, site_id, model_data, model_delta_meta):
        cluster = self.get_cluster_for_site(site_id)
        with self.cluster_locks[cluster]:
            self.cluster_models[cluster] = self.aggregator.aggregate(self.cluster_models[cluster], model_data, model_delta_meta)
    
    def update_global_model(self, model_data, model_delta_meta):
        with self.global_model_lock:
            self.global_model = self.aggregator.aggregate(self.global_model, model_data, model_delta_meta)

    def get_cluster_model(self, site_id):
        cluster = self.get_cluster_for_site(site_id)
        with self.cluster_locks[cluster]:
            return self.cluster_models[cluster]
        
    def get_global_model(self):
        with self.global_model_lock:
            return self.global_model



