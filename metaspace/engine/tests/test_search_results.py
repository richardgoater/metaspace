import json
from collections import OrderedDict
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
import numpy as np
from scipy.sparse import coo_matrix

from sm.engine.db import DB
from sm.engine.ion_mapping import ION_SEL
from sm.engine.image_store import ImageStoreServiceWrapper
from sm.engine.search_results import SearchResults, METRICS_INS, post_images_to_image_store

db_mock = MagicMock(spec=DB)


def db_sel_side_effect(query, *args):
    if query == ION_SEL:
        # formula, chem_mod, neutral_loss, adduct, id
        return [('H2O', '', '-OH', '+H', 123)]
    raise ValueError(f'Unrecognized db.select: {query} {args}')


@pytest.fixture
def search_results():
    metrics = [
        'chaos',
        'spatial',
        'spectral',
        'msm',
        'total_iso_ints',
        'min_iso_ints',
        'max_iso_ints',
    ]
    res = SearchResults(0, metrics, 4, 1)
    return res


def _mock_ion_metrics_df():
    return pd.DataFrame(
        [
            (
                13,
                'H2O',
                '-OH+H',
                '+H',
                '',
                '-OH',
                0.9,
                0.9,
                0.9,
                0.9 ** 3,
                [100, 10],
                [0, 0],
                [10, 1],
                0.5,
            )
        ],
        columns=[
            'formula_i',
            'formula',
            'modifier',
            'adduct',
            'chem_mod',
            'neutral_loss',
            'chaos',
            'spatial',
            'spectral',
            'msm',
            'total_iso_ints',
            'min_iso_ints',
            'max_iso_ints',
            'fdr',
        ],
    ).set_index('formula_i')


def test_save_ion_img_metrics_correct_db_call(search_results):
    ion_img_ids = {13: {'iso_image_ids': ['iso_image_1', None, None, None]}}
    ion_metrics_df = _mock_ion_metrics_df()
    db_mock.select.side_effect = db_sel_side_effect

    search_results.store_ion_metrics(ion_metrics_df, ion_img_ids, db_mock)

    metrics_json = json.dumps(
        OrderedDict(
            zip(
                [
                    'chaos',
                    'spatial',
                    'spectral',
                    'msm',
                    'total_iso_ints',
                    'min_iso_ints',
                    'max_iso_ints',
                ],
                (0.9, 0.9, 0.9, 0.9 ** 3, [100, 10], [0, 0], [10, 1]),
            )
        )
    )
    exp_rows = [
        (
            0,
            'H2O',
            '',
            '-OH',
            '+H',
            0.9 ** 3,
            0.5,
            metrics_json,
            ['iso_image_1', None, None, None],
            123,
        )
    ]
    db_mock.insert.assert_called_with(METRICS_INS, exp_rows)


def test_isotope_images_are_stored(search_results, pysparkling_context):
    mask = np.array([[1, 1], [1, 0]])
    img_id = "iso_image_id"
    img_store_mock = Mock(spec=ImageStoreServiceWrapper)
    img_store_mock.post_image.return_value = img_id

    img_store_mock.reset_mock()
    formula_images_rdd = pysparkling_context.parallelize(
        [
            (0, [coo_matrix([[0, 0], [0, 1]]), None, coo_matrix([[2, 3], [1, 0]]), None]),
            (1, [coo_matrix([[1, 1], [0, 1]]), None, None, None]),
        ]
    )
    ids = post_images_to_image_store(formula_images_rdd, mask, img_store_mock, 'fs', 4)
    assert ids == {
        0: {'iso_image_ids': [img_id, None, img_id, None]},
        1: {'iso_image_ids': [img_id, None, None, None]},
    }
    assert img_store_mock.post_image.call_count == 3


def test_non_native_python_number_types_handled(search_results):
    ion_img_ids = {13: {'iso_image_ids': ['iso_image_1', None, None, None]}}
    metrics_df = _mock_ion_metrics_df()
    db_mock.select.side_effect = db_sel_side_effect

    for col in ['chaos', 'spatial', 'spectral', 'msm', 'fdr']:
        metrics_df[col] = metrics_df[col].astype(np.float64)

        search_results.store_ion_metrics(metrics_df, ion_img_ids, db_mock)

        metrics_json = json.dumps(
            OrderedDict(
                zip(
                    [
                        'chaos',
                        'spatial',
                        'spectral',
                        'msm',
                        'total_iso_ints',
                        'min_iso_ints',
                        'max_iso_ints',
                    ],
                    (0.9, 0.9, 0.9, 0.9 ** 3, [100, 10], [0, 0], [10, 1]),
                )
            )
        )
        exp_rows = [
            (
                0,
                'H2O',
                '',
                '-OH',
                '+H',
                0.9 ** 3,
                0.5,
                metrics_json,
                ['iso_image_1', None, None, None],
                123,
            )
        ]
        db_mock.insert.assert_called_with(METRICS_INS, exp_rows)


def test_save_ion_img_metrics_empty_call(search_results):
    ion_img_ids = {}
    ion_metrics_df = _mock_ion_metrics_df().iloc[0:0]
    db_mock.select.side_effect = db_sel_side_effect

    search_results.store_ion_metrics(ion_metrics_df, ion_img_ids, db_mock)

    db_mock.insert.assert_called_with(METRICS_INS, [])
