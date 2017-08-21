from app.database import db_session, init_session
from app.log import get_logger
from app.model.molecule import Molecule

import pathlib
import pyarrow.parquet
import pandas as pd
import cpyMSpec as ms

logger = get_logger()

ISOTOPIC_PEAK_N = 4
SIGMA_TO_FWHM = 2.3548200450309493  # 2 \sqrt{2 \log 2}

class InstrumentSettings(object):
    def __init__(self, pts_per_mz):
        self.pts_per_mz = pts_per_mz


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

    def _molecular_formulas(self, db_id):
        if db_id in self._mf_cache:
            return self._mf_cache[db_id]

        mf_tuples = self.db_session.query(Molecule.sf)\
                                   .filter(Molecule.db_id == db_id)\
                                   .distinct('sf').all()
        self._mf_cache[db_id] = {t[0] for t in mf_tuples}
        return self._mf_cache[db_id]

    def load_patterns(self, db_id, instrument_settings, charge):
        mol_formulas = self._molecular_formulas(db_id)
        dir_path = self._dir_path(instrument_settings, charge)
        df = self._load(dir_path)
        return df[df['mf'].isin(mol_formulas)]

    def add_database(self, db_id, instrument_settings, adduct, charge):
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


DECOY_ADDUCTS = ['+He', '+Li', '+Be', '+B', '+C', '+N', '+O', '+F', '+Ne', '+Mg', '+Al', '+Si', '+P', '+S', '+Cl', '+Ar', '+Ca', '+Sc', '+Ti', '+V', '+Cr', '+Mn', '+Fe', '+Co', '+Ni', '+Cu', '+Zn', '+Ga', '+Ge', '+As', '+Se', '+Br', '+Kr', '+Rb', '+Sr', '+Y', '+Zr', '+Nb', '+Mo', '+Ru', '+Rh', '+Pd', '+Ag', '+Cd', '+In', '+Sn', '+Sb', '+Te', '+I', '+Xe', '+Cs', '+Ba', '+La', '+Ce', '+Pr', '+Nd', '+Sm', '+Eu', '+Gd', '+Tb', '+Dy', '+Ho', '+Ir', '+Th', '+Pt', '+Os', '+Yb', '+Lu', '+Bi', '+Pb', '+Re', '+Tl', '+Tm', '+U', '+W', '+Au', '+Er', '+Hf', '+Hg', '+Ta']


if __name__ == '__main__':
    init_session()
    settings = InstrumentSettings(4039)  # corresponds to 140K

    pattern_storage = IsotopePatternStorage(db_session, "/tmp/isotope_patterns")

    for db_id in [2, 3, 4]:
        configurations = []

        for adduct in ['+H', '+K', '+Na']:
            configurations.append((adduct, 1))

        for adduct in ['-H', '+Cl']:
            configurations.append((adduct, -1))

        for adduct in DECOY_ADDUCTS:
            configurations.append((adduct, 1))
            configurations.append((adduct, -1))

        def job(configuration):
            adduct, charge = configuration
            pattern_storage.add_database(db_id, settings, adduct, charge)

        for conf in configurations:
            job(conf)
