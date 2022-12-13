from pathlib import Path
from ast import literal_eval
import pickle

import pandas as pd


class UserKNN:

    def __init__(self, model_path: Path, interactions_path: Path,
                 users_mapping: Path, users_inv_mapping: Path):
        self.model, self.dataset, self.users_mapping, self.users_inv_mapping = \
            self._load_saved_models(
                model_path=model_path,
                interactions_path=interactions_path,
                users_mapping=users_mapping,
                users_inv_mapping=users_inv_mapping
            )
        # self.train_dataset = self._load_train_dataset(dataset_path)
        self.mapper = self._generate_implicit_recs_mapper(
            model=self.model,
            n_recs=30,
            users_mapping=self.users_mapping,
            users_inv_mapping=self.users_inv_mapping
        )

    def predict(self, user_id_list: list):
        recs = pd.DataFrame({
            'user_id': user_id_list
        })

        recs['similar_user_id'], recs['similarity'] = zip(
            *recs['user_id'].map(self.mapper)
        )
        recs = recs.set_index('user_id').apply(pd.Series.explode).reset_index()
        recs = recs.sort_values(['user_id', 'similarity'], ascending=False)
        recs = recs.merge(self.dataset, left_on=['similar_user_id'],
                          right_on=['user_id'], how='left')
        recs = recs.explode('item_id')
        recs = recs.sort_values(['user_id', 'similarity'], ascending=False)

        recs = recs.drop_duplicates(['user_id', 'item_id'], keep='first')

        return recs['item_id'].tolist()

    @staticmethod
    def _load_saved_models(model_path: Path, interactions_path: Path,
                           users_mapping: Path, users_inv_mapping: Path):
        if model_path.is_file():
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
        else:
            raise FileNotFoundError(f'Model {model_path} was not found')
        if interactions_path.is_file():
            dataset = pd.read_csv(interactions_path)
            dataset = dataset.groupby('user_id').agg({'item_id': list})
        else:
            raise FileNotFoundError(f'Dataframe {model_path} was not found')
        if users_mapping.is_file():
            with open(users_mapping, 'rb') as users_m_file:
                users_mapping_file = pickle.load(users_m_file)
        else:
            raise FileNotFoundError(f'Mapping {users_mapping} was not found')
        if users_inv_mapping.is_file():
            with open(users_inv_mapping, 'rb') as users_inv_m_file:
                users_inv_mapping_file = pickle.load(users_inv_m_file)
        else:
            raise FileNotFoundError(
                f'Mapping {users_inv_mapping} was not found')
        return model, dataset, users_mapping_file, users_inv_mapping_file

    # @staticmethod
    # def _load_train_dataset(dataset_path: Path):
    #
    #     if dataset_path.is_file():
    #         dataset = pd.read_csv(dataset_path)
    #         dataset.rename(columns={'last_watch_dt': 'datetime',
    #                                 'total_dur': 'weight'},
    #                        inplace=True)
    #         dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    #         dataset['rank'] = dataset.groupby('user_id').cumcount() + 1
    #         dataset = dataset[dataset['rank'] <= 10]
    #         return dict(zip(dataset.loc[:, 'user_id'], dataset.loc[:, 'item_id']))
    #     else:
    #         raise FileNotFoundError(f'Dataset {dataset_path} was not found.')

    @staticmethod
    def _generate_implicit_recs_mapper(model, n_recs, users_mapping,
                                       users_inv_mapping):
        def _recs_mapper(user):
            user_id = users_mapping[user]
            recs = model.similar_items(user_id, N=n_recs)
            return [users_inv_mapping[user] for user, _ in recs],\
                   [sim for _, sim in recs]

        return _recs_mapper

    @classmethod
    def from_config(cls, config):

        model_path = Path(literal_eval(
            config['UserKNN']['model_path'])
        )
        interactions_path = Path(literal_eval(
            config['Datasets']['interactions_path'])
        )
        users_mapping = Path(literal_eval(
            config['UserKNN']['users_mapping'])
        )
        users_inv_mapping = Path(literal_eval(
            config['UserKNN']['users_inv_mapping'])
        )

        return cls(
            model_path=model_path,
            interactions_path=interactions_path,
            users_mapping=users_mapping,
            users_inv_mapping=users_inv_mapping
        )
