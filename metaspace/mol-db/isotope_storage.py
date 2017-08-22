from app.database import db_session, init_session
from app.log import get_logger
from app.model.molecule import Molecule

import pyarrow.parquet
import pandas as pd
import cpyMSpec as ms

from itertools import product
import pathlib

logger = get_logger()

ISOTOPIC_PEAK_N = 4
SIGMA_TO_FWHM = 2.3548200450309493  # 2 \sqrt{2 \log 2}

class InstrumentSettings(object):
    def __init__(self, pts_per_mz):
        self.pts_per_mz = pts_per_mz

    def __repr__(self):
        return "<Instrument(pts_per_mz={})>".format(self.pts_per_mz)


class IsotopePatternStorage(object):
    def __init__(self, db_session, directory):
        self.db_session = db_session
        self._dir = pathlib.Path(directory)
        self._dir.mkdir(exist_ok=True, parents=True)
        self._mf_cache = {}

    def _dir_path(self, instrument_settings, charge):
        return self._dir / "pts_{}".format(instrument_settings.pts_per_mz) / "charge_{}".format(charge)

    def _path(self, instrument_settings, adduct, charge):
        dir_path = self._dir_path(instrument_settings, charge)
        full_path = dir_path / "adduct_{}.parquet".format(adduct)
        return full_path

    def _load(self, filename):
        return pyarrow.parquet.read_table(str(filename)).to_pandas()

    def _molecular_formulas(self, db_id=None):
        """
        db_id=None means all databases
        """
        if db_id is None:
            db_id = -1

        if db_id in self._mf_cache:
            return self._mf_cache[db_id]

        q = self.db_session.query(Molecule.sf)
        if db_id != -1:
            q = q.filter(Molecule.db_id == db_id)
        mf_tuples = q.distinct('sf').all()
        self._mf_cache[db_id] = {t[0] for t in mf_tuples}
        return self._mf_cache[db_id]

    def _configurations(self):
        """
        Represents directory tree as a dataframe with pts_per_mz, charge, adduct columns
        """
        result = []
        for instr_dir in self._dir.iterdir():
            pts_per_mz = int(instr_dir.name.split('_')[1])
            for charge_dir in instr_dir.iterdir():
                charge = int(charge_dir.name.split('_')[1])
                for adduct_dir in charge_dir.iterdir():
                    adduct = adduct_dir.name.split('_')[1].split('.')[0]
                    result.append(dict(pts_per_mz=pts_per_mz, charge=charge, adduct=adduct))

        return pd.DataFrame.from_records(result)

    def adduct_charge_pairs(self):
        """
        Set of all (adduct, charge) combinations added so far
        """
        return {(x[1], x[2]) for x in self._configurations()[['adduct', 'charge']].itertuples()}

    def instrument_settings(self):
        """
        Set of all instrument settings added so far
        """
        return {InstrumentSettings(pts_per_mz=x) for x in self._configurations()['pts_per_mz'].unique()}

    def load_patterns(self, instrument_settings, charge, db_id=None):
        dir_path = self._dir_path(instrument_settings, charge)
        df = self._load(dir_path)
        if not db_id:  # = all databases
            return df
        mol_formulas = self._molecular_formulas(db_id)
        return df[df['mf'].isin(mol_formulas)]

    def generate_patterns(self, instrument_settings, adduct, charge, db_id=None):
        """
        Fine-grained function for generating/updating a single file.
        db_id allows selecting a particular database, None value means all databases
        """
        dump_path = self._path(instrument_settings, adduct, charge)

        existing_df = None
        mf_finished = set()
        fn = str(dump_path)
        if dump_path.exists():
            existing_df = self._load(fn)
            mf_finished = set(existing_df['mf'])
            logger.info('read {} isotope patterns from {}'.format(len(mf_finished), fn))
        else:
            dump_path.parent.mkdir(exist_ok=True, parents=True)

        new_mol_formulas = list(self._molecular_formulas(db_id) - mf_finished)

        if not new_mol_formulas:
            logger.info('no new molecular formulas detected')
            return

        valid_mfs = []
        masses = []
        intensities = []
        for mf in new_mol_formulas:
            try:
                p = ms.isotopePattern(mf + adduct)
            except Exception as e:
                # logger.warning("skipped {}".format(mf) + ': ' + str(e))
                continue

            valid_mfs.append(mf)

            sigma = 5.0 / instrument_settings.pts_per_mz
            fwhm = sigma * SIGMA_TO_FWHM
            resolving_power = p.masses[0] / fwhm
            instrument_model = ms.InstrumentModel('tof', resolving_power)

            p.addCharge(charge)
            p = p.centroids(instrument_model).trimmed(ISOTOPIC_PEAK_N)
            p.sortByMass()
            masses.append(p.masses)
            intensities.append(p.intensities)

        df = pd.DataFrame({
            'mf': valid_mfs,
            'mzs': masses,
            'intensities': intensities
        })
        df['adduct'] = adduct
        if existing_df is not None and not existing_df.empty:
            df = pd.concat([existing_df, df])

        table = pyarrow.Table.from_pandas(df)
        pyarrow.parquet.write_table(table, fn)

        logger.info('wrote {} NEW isotope patterns to {}'.format(len(new_mol_formulas), fn))

    def batch_generate(self,
                       database_ids=None,
                       instrument_settings_list=None,
                       adduct_charge_pairs=None):
        """
        Generate isotope patterns for cartesian product of configurations
        Each parameter is either a list or None where the latter means taking all existing values.
        """
        param_tuples = product(database_ids or [None],
                               instrument_settings_list or self.instrument_settings(),
                               adduct_charge_pairs or self.adduct_charge_pairs())
        for params in param_tuples:
            adduct, charge = params[-1]
            db_id, instrument_settings = params[0:2]
            self.generate_patterns(instrument_settings, adduct, charge, db_id)


DECOY_ADDUCTS = ['+He', '+Li', '+Be', '+B', '+C', '+N', '+O', '+F', '+Ne', '+Mg', '+Al', '+Si', '+P', '+S', '+Cl', '+Ar', '+Ca', '+Sc', '+Ti', '+V', '+Cr', '+Mn', '+Fe', '+Co', '+Ni', '+Cu', '+Zn', '+Ga', '+Ge', '+As', '+Se', '+Br', '+Kr', '+Rb', '+Sr', '+Y', '+Zr', '+Nb', '+Mo', '+Ru', '+Rh', '+Pd', '+Ag', '+Cd', '+In', '+Sn', '+Sb', '+Te', '+I', '+Xe', '+Cs', '+Ba', '+La', '+Ce', '+Pr', '+Nd', '+Sm', '+Eu', '+Gd', '+Tb', '+Dy', '+Ho', '+Ir', '+Th', '+Pt', '+Os', '+Yb', '+Lu', '+Bi', '+Pb', '+Re', '+Tl', '+Tm', '+U', '+W', '+Au', '+Er', '+Hf', '+Hg', '+Ta']


if __name__ == '__main__':
    init_session()
    settings = InstrumentSettings(4039)  # corresponds to 140K

    pattern_storage = IsotopePatternStorage(db_session, "/tmp/isotope_patterns")

    adduct_charge_pairs = []

    for adduct in ['+H', '+K', '+Na']:
        adduct_charge_pairs.append((adduct, 1))

    for adduct in ['-H', '+Cl']:
        adduct_charge_pairs.append((adduct, -1))

    for adduct in DECOY_ADDUCTS:
        adduct_charge_pairs.append((adduct, 1))
        adduct_charge_pairs.append((adduct, -1))

    pattern_storage.batch_generate(instrument_settings_list=[settings],
                                   adduct_charge_pairs=adduct_charge_pairs)
