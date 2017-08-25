import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[1]))

from app.isotope_storage import IsotopePatternStorage, InstrumentSettings

import pytest

import shutil
import tempfile

@pytest.fixture(scope='module')
def molecular_formula_sets():
    return {
        1: {'H2O', 'C3H6O3', 'CO2'},
        2: {'H2O', 'N2', 'C2H5OH'},
        3: {'CO2', 'C2H4'}
    }

@pytest.fixture
def storage_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    #shutil.rmtree(dir_path)

INSTR = InstrumentSettings(pts_per_mz=5000)

@pytest.fixture
def storage(molecular_formula_sets, storage_dir):
    return IsotopePatternStorage(molecular_formula_sets, storage_dir)

def test_generate_load_all_sets(storage):
    storage.generate_patterns(INSTR, '+H', 1)
    storage.generate_patterns(INSTR, '-H', -1)
    df_pos = storage.load_patterns(INSTR, charge=1, db_id=None, adducts=['+H'])
    df_neg = storage.load_patterns(INSTR, charge=-1, db_id=None, adducts=['-H'])
    assert len(df_pos) == 6  # number of unique molecular formulas in all sets
    assert len(df_neg) == 4  # CO2 & N2 don't have H => only 4 remain

def test_generate_load_one_set(storage):
    storage.generate_patterns(INSTR, '+H', 1)
    storage.generate_patterns(INSTR, '-H', -1)
    df_pos = storage.load_patterns(INSTR, charge=1, db_id=1, adducts=['+H'])
    df_neg = storage.load_patterns(INSTR, charge=-1, db_id=1, adducts=['-H'])
    assert len(df_pos) == 3  # size of the set with id = 1
    assert len(df_neg) == 2  # CO2 doesn't have H thus one less

def test_batch_generate(storage):
    storage.batch_generate(database_ids=None, instrument_settings_list=[INSTR],
                           adduct_charge_pairs=[('+H', 1), ('+Na', 1), ('-H', -1), ('+Cl', -1)],
                           n_processes=1)
    df_pos = storage.load_patterns(INSTR, charge=1, db_id=None, adducts=['+H', '+Na'])
    assert len(df_pos) == 6 * 2
    df_neg = storage.load_patterns(INSTR, charge=-1, db_id=None, adducts=['-H', '+Cl'])
    assert len(df_neg) == 4 + 6  # 4 for -H, all 6 for +Cl
    df_empty = storage.load_patterns(INSTR, charge=1, db_id=None, adducts=['-H', '+Cl'])
    assert df_empty.empty

def test_load_patterns(storage):
    storage.batch_generate(database_ids=None, instrument_settings_list=[INSTR],
                           adduct_charge_pairs=[('+H', 1), ('+Na', 1), ('-H', -1), ('+Cl', -1)],
                           n_processes=1)
    assert 3 == len(storage.load_patterns(INSTR, charge=1, db_id=1, adducts=['+H']))
    assert 1 == len(storage.load_patterns(INSTR, charge=-1, db_id=3, adducts=['-H']))
    assert 6 == len(storage.load_patterns(INSTR, charge=1, db_id=2, adducts=['+H', '+Na']))
    assert 6 == len(storage.load_patterns(INSTR, charge=-1, db_id=None, adducts=['+Cl']))
    assert 4 == len(storage.load_patterns(INSTR, charge=-1, db_id=None, adducts=['-H']))
    assert 0 == len(storage.load_patterns(INSTR, charge=1, db_id=None, adducts=['-H']))

def test_fdr_subsampling(storage):
    targets = ['+H', '+Na']
    decoys = ['+Fe', '+Ne', '+Mn', '+W', '+Ar']
    for adduct in targets + decoys:
        storage.generate_patterns(INSTR, adduct, 1)
    n_decoys_per_target = 3
    df = storage.load_fdr_subsample(INSTR, 1, 1, targets, decoys, n_decoys_per_target)
    n_mfs = len(df['mf'].unique())
    assert df['+H'].sum() == n_mfs * n_decoys_per_target
    assert df['+Na'].sum() == n_mfs * n_decoys_per_target
    assert df['is_target'].sum() == n_mfs * len(targets)
    for mf, group in df.groupby('mf'):
        assert group['+H'].sum() == n_decoys_per_target
        assert group['+Na'].sum() == n_decoys_per_target
        assert len(group) >= n_decoys_per_target + len(targets)
