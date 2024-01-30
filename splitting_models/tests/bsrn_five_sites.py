
import time
import tempfile
import functools
from pathlib import Path

import pandas as pd
from loguru import logger

import splitting_models.models as sm


logger.enable(__name__)


@functools.cache
def load_valid_data():
    temp_data_file = Path(tempfile.gettempdir()) / 'bsrn_valid_five_sites.parquet'
    if not temp_data_file.exists():
        logger.info(f'downloading data from remote dataset. Saving to {temp_data_file}')
        REMOTE_FILE_URL = (
            'https://zenodo.org/records/10593079/files/'
            'bsrn_valid_five_sites.parquet?download=1')
        pd.read_parquet(REMOTE_FILE_URL).to_parquet(temp_data_file)
    return pd.read_parquet(temp_data_file)


def test(models=None):
    data = load_valid_data()
    figs = []
    error_metrics = pd.DataFrame()
    for model_name in ('Erbs', 'Perez2', 'Engerer2', 'Starke3', 'Yang5', 'GISPLIT'):
        logger.info(f'Running the {model_name} model')
        c1 = time.perf_counter()
        model_obj = getattr(sm, model_name)()
        fig, em = model_obj.diagnostics(
            data, return_error_metrics=True,
            sky_type_or_func=lambda df: df.sky_type.values)
        logger.info(f'  exec. time: {time.perf_counter() - c1:.2f} seconds')
        error_metrics = pd.concat([error_metrics, em], axis=0)
        figs.append(fig)
    print(error_metrics)
    return figs
