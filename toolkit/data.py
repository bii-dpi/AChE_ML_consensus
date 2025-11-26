import joblib

from . import config


def get_split_input_data(structure=None, score_types=None):
    if isinstance(score_types, list):
        output = joblib.load(f"data/processed/both.pkl")
    else:
        output = joblib.load(f"data/processed/{score_types}.pkl")

    if structure is not None:
        assert not isinstance(structure, list)

        index = config.PDB_IDS.index(structure)
        if isinstance(score_types, list):
            index *= 2

            output[0][0]= output[0][0][:, index:index + 2]
            output[1][0] = output[1][0][:, index:index + 2]
        else:
            output[0][0]= output[0][0][:, index:index + 1]
            output[1][0] = output[1][0][:, index:index + 1]

    return output


