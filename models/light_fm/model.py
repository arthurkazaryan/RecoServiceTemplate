import pickle
from pathlib import Path
from ast import literal_eval

import numpy as np


class LightFM:

    def __init__(
        self,
        model_path: Path,
        users_mapping_path: Path,
        items_inv_mapping_path: Path
    ):
        self.model, self.users_mapping, self.items_inv_mapping = \
            self._load_saved_models(
                model_path=model_path,
                users_mapping_path=users_mapping_path,
                items_inv_mapping_path=items_inv_mapping_path
            )
        self.items_biases, self.items_embedding = self.model.get_item_representations()

    def predict(self, user_id: int, n_recs: int):

        most_popular = [9728, 10440, 15297, 13865, 14488,
                        12192, 12360, 341, 4151, 3734]
        if user_id in self.users_mapping:
            user_inner_idx = self.users_mapping[user_id]
            user_biases, user_embedding = \
                self.model.get_user_representations()[0][user_inner_idx],\
                self.model.get_user_representations()[1][user_inner_idx]

            user_embedding = np.hstack(
                (user_biases, np.ones(user_biases.size), user_embedding)
            )
            items_embedding = np.hstack(
                (np.ones((self.items_biases.size, 1)),
                 self.items_biases[:, np.newaxis],
                 self.items_embedding)
            )

            scores = items_embedding @ user_embedding
            top_score_ids = scores.argsort()[-n_recs:][::-1]
            recommendations = \
                [self.items_inv_mapping[item] for item in top_score_ids]
        else:
            recommendations = most_popular

        return recommendations

    @staticmethod
    def _load_saved_models(
        model_path: Path,
        users_mapping_path: Path,
        items_inv_mapping_path: Path
    ):
        if model_path.is_file():
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
        else:
            raise FileNotFoundError(f'Model {model_path} was not found')

        if users_mapping_path.is_file():
            with open(users_mapping_path, 'rb') as model_file:
                users_mapping = pickle.load(model_file)
        else:
            raise FileNotFoundError(f'Users mapping data was not found')

        if items_inv_mapping_path.is_file():
            with open(items_inv_mapping_path, 'rb') as model_file:
                items_inv_mapping = pickle.load(model_file)
        else:
            raise FileNotFoundError(f'Items inv mapping was not found')

        return model, users_mapping, items_inv_mapping

    @classmethod
    def from_config(cls, config):

        model_path = Path(literal_eval(
            config['LightFM']['model_path']
        ))
        users_mapping_path = Path(literal_eval(
            config['LightFM']['users_mapping']
        ))
        items_inv_mapping_path = Path(literal_eval(
            config['LightFM']['items_inv_mapping']
        ))

        return cls(
            model_path=model_path,
            users_mapping_path=users_mapping_path,
            items_inv_mapping_path=items_inv_mapping_path
        )
