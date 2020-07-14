import json
import os
import pickle
from collections import OrderedDict
import logging
from datetime import datetime
from pathlib import Path
import uuid

from sm.engine.ion_mapping import get_ion_id_mapping
from sm.engine.png_generator import PngGenerator

logger = logging.getLogger('engine')
METRICS_INS = (
    'INSERT INTO annotation ('
    '   job_id, formula, chem_mod, neutral_loss, adduct, msm, fdr, stats, iso_image_ids, ion_id'
    ') '
    'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
)


def post_images_to_image_store(
    formula_images_rdd, alpha_channel, img_store, img_store_type, n_peaks
):
    # logger.info(f'Posting iso images to {img_store}')
    # png_generator = PngGenerator(alpha_channel, greyscale=True)
    logger.info(f'Dumping images')
    p = Path('/tmp/iso_images') / datetime.utcnow().isoformat()

    def generate_png_and_post(partition_idx, kvs):
        result = []
        images = {}
        for k, imgs in kvs:
            iso_image_ids = [None] * n_peaks
            for peak_i, img in enumerate(imgs):
                if img is not None:
                    iso_image_ids[peak_i] = img_id = str(uuid.uuid4())
                    images[img_id] = img
                    # fp = png_generator.generate_png(img.toarray())
                    # iso_image_ids[k] = img_store.post_image(img_store_type, 'iso_image', fp)
            result.append((k, {'iso_image_ids': iso_image_ids}))
        p.mkdir(parents=True, exist_ok=True)
        pickle.dump(images, (p / f'images_{partition_idx}.pickle').open('wb'))

        return result

    result = dict(formula_images_rdd.mapPartitionsWithIndex(generate_png_and_post).collect())
    n_images = len([img for imgs in result.values() for img in imgs if img is not None])
    logger.info(f'Saved images: {n_images} images')
    return result


class SearchResults:
    """ Container for molecule search results

    Args
    ----------
    job_id : int
        Search job id
    metric_names: list
        Metric names
    """

    def __init__(self, job_id, metric_names, n_peaks, charge):
        self.job_id = job_id
        self.metric_names = metric_names
        self.n_peaks = n_peaks
        self.charge = charge

    def _metrics_table_row_gen(self, job_id, metr_df, formula_img_ids, ion_mapping):
        for _, row in metr_df.iterrows():
            m = OrderedDict((name, row[name]) for name in self.metric_names)
            metr_json = json.dumps(m)
            image_ids = formula_img_ids[row.formula_i]['iso_image_ids']
            yield (
                job_id,
                row.formula,
                row.chem_mod,
                row.neutral_loss,
                row.adduct,
                float(row.msm),
                float(row.fdr),
                metr_json,
                image_ids,
                ion_mapping[(row.formula), (row.chem_mod), (row.neutral_loss), (row.adduct)],
            )

    def store_ion_metrics(self, ion_metrics_df, ion_img_ids, db):
        """ Store formula image metrics and image ids in the database """
        logger.info('Storing iso image metrics')

        ions = ion_metrics_df[['formula', 'chem_mod', 'neutral_loss', 'adduct']]
        ion_tuples = list(ions.itertuples(False, None))
        ion_mapping = get_ion_id_mapping(db, ion_tuples, self.charge)

        rows = list(
            self._metrics_table_row_gen(
                self.job_id, ion_metrics_df.reset_index(), ion_img_ids, ion_mapping
            )
        )
        logger.info(f'Saving {len(rows)} metrics')
        p = Path('/tmp/search_results')
        p.mkdir(parents=True, exist_ok=True)
        pickle.dump(
            list(rows), (p / f'{datetime.utcnow().isoformat()}_{self.job_id}.pickle').open('wb')
        )

        # db.insert(METRICS_INS, list(rows))

    def store(self, metrics_df, formula_images_rdd, alpha_channel, db, img_store, img_store_type):
        """ Save formula metrics and images

        Args
        ---------
        metrics_df : pandas.Dataframe
            formula, adduct, msm, fdr, individual metrics
        formula_images_rdd : pyspark.RDD
            values must be lists of 2d intensity arrays (in coo_matrix format)
        alpha_channel : numpy.array
            Image alpha channel (2D, 0..1)
        db : sm.engine.DB
            database connection
        img_store : sm.engine.png_generator.ImageStoreServiceWrapper
            m/z image store
        img_store_type: str
        """
        logger.info('Storing search results to the DB')
        formula_image_ids = post_images_to_image_store(
            formula_images_rdd, alpha_channel, img_store, img_store_type, self.n_peaks
        )
        self.store_ion_metrics(metrics_df, formula_image_ids, db)
