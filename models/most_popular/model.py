import pickle
from pathlib import Path
from ast import literal_eval

import pandas as pd
import numpy as np
from rectools.dataset import Dataset
from rectools.dataset.identifiers import IdMap


class MostPopular:

    def __init__(
        self,
        model_path: Path,
        interactions_df_path: Path,
        item_features_df_path: Path
    ):
        self.model, self.dataset = self._load_saved_models(
                model_path=model_path,
                interactions_df_path=interactions_df_path,
                item_features_df_path=item_features_df_path
            )

    def predict(self, user_id: int, n_recs: int):

        recommendations = self.model.recommend(
            self.dataset.user_id_map.external_ids[:1],
            dataset=self.dataset,
            k=n_recs,
            filter_viewed=False
        )

        return recommendations['item_id'].tolist()

    @staticmethod
    def _load_saved_models(
        model_path: Path,
        interactions_df_path: Path,
        item_features_df_path: Path
    ):
        if model_path.is_file():
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
        else:
            raise FileNotFoundError(f'Model {model_path} was not found')

        if interactions_df_path.is_file() and item_features_df_path.is_file():
            dataset = Dataset.construct(
                interactions_df=pd.read_csv(interactions_df_path),
                user_features_df=None,
                item_features_df=pd.read_csv(item_features_df_path),
                cat_item_features=['genre', 'release_year']
            )
        else:
            raise FileNotFoundError(f'CSV data was not found')

        return model, dataset

    @classmethod
    def from_config(cls, config):

        model_path = Path(literal_eval(
            config['MostPopular']['model_path']
        ))
        interactions_df_path = Path(literal_eval(
            config['MostPopular']['interactions']
        ))
        item_features_df_path = Path(literal_eval(
            config['MostPopular']['item_features']
        ))

        return cls(
            model_path=model_path,
            interactions_df_path=interactions_df_path,
            item_features_df_path=item_features_df_path
        )
