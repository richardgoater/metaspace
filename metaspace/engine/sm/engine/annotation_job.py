import time
from pathlib import Path
from pprint import pformat
from datetime import datetime
from shutil import copytree, rmtree
import logging

import boto3
from pyspark import SparkContext, SparkConf

from sm.engine.acq_geometry import make_acq_geometry
from sm.engine.imzml_parser import ImzMLParserWrapper
from sm.engine.msm_basic.formula_imager import make_sample_area_mask, get_ds_dims
from sm.engine.msm_basic.formula_validator import METRICS
from sm.engine.msm_basic.msm_basic_search import MSMSearch
from sm.engine.db import DB
from sm.engine.search_results import SearchResults
from sm.engine.util import SMConfig, split_s3_path
from sm.engine.es_export import ESExporter
from sm.engine import molecular_db
from sm.engine.queue import QueuePublisher, SM_DS_STATUS

logger = logging.getLogger('engine')

JOB_ID_MOLDB_ID_SEL = "SELECT id, moldb_id FROM job WHERE ds_id = %s AND status='FINISHED'"
JOB_INS = "INSERT INTO job (moldb_id, ds_id, status, start) VALUES (%s, %s, %s, %s) RETURNING id"
JOB_UPD_STATUS_FINISH = "UPDATE job set status=%s, finish=%s where id=%s"
JOB_UPD_FINISH = "UPDATE job set finish=%s where id=%s"
TARGET_DECOY_ADD_DEL = (
    'DELETE FROM target_decoy_add tda WHERE tda.job_id IN (SELECT id FROM job WHERE ds_id = %s)'
)


class JobStatus:
    RUNNING = 'RUNNING'
    FINISHED = 'FINISHED'
    FAILED = 'FAILED'


class AnnotationJob:
    """Class responsible for dataset annotation."""

    def __init__(self, img_store=None, sm_config=None):
        self._img_store = img_store

        self._sc = None
        self._db = DB()
        self._ds = None
        self._status_queue = None
        self._es = None

        self._sm_config = sm_config or SMConfig.get_conf()
        self._ds_data_path = None

    def _configure_spark(self):
        logger.info('Configuring Spark')
        sconf = SparkConf()
        for prop, value in self._sm_config['spark'].items():
            if prop.startswith('spark.'):
                sconf.set(prop, value)

        if 'aws' in self._sm_config:
            sconf.set("spark.hadoop.fs.s3a.access.key", self._sm_config['aws']['aws_access_key_id'])
            sconf.set(
                "spark.hadoop.fs.s3a.secret.key", self._sm_config['aws']['aws_secret_access_key']
            )
            sconf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            sconf.set(
                "spark.hadoop.fs.s3a.endpoint",
                "s3.{}.amazonaws.com".format(self._sm_config['aws']['aws_default_region']),
            )

        self._sc = SparkContext(
            master=self._sm_config['spark']['master'], conf=sconf, appName='SM engine'
        )

    def _store_job_meta(self, moldb_id: int):
        """Store search job metadata in the database."""

        logger.info('Storing job metadata')
        rows = [
            (
                moldb_id,
                self._ds.id,
                JobStatus.RUNNING,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        ]
        return self._db.insert_return(JOB_INS, rows=rows)[0]

    def create_imzml_parser(self):
        logger.info('Parsing imzml')
        return ImzMLParserWrapper(self._ds_data_path)

    def _run_annotation_jobs(self, imzml_parser, moldbs):
        if moldbs:
            logger.info(
                f"Running new job ds_id: {self._ds.id}, ds_name: {self._ds.name}, mol dbs: {moldbs}"
            )

            # FIXME: Total runtime of the dataset should be measured, not separate jobs
            job_ids = [self._store_job_meta(moldb.id) for moldb in moldbs]

            search_alg = MSMSearch(
                spark_context=self._sc,
                imzml_parser=imzml_parser,
                moldbs=moldbs,
                ds_config=self._ds.config,
                ds_data_path=self._ds_data_path,
            )
            search_results_it = search_alg.search()

            for job_id, (moldb_ion_metrics_df, moldb_ion_images_rdd) in zip(
                job_ids, search_results_it
            ):
                # Save results for each moldb
                job_status = JobStatus.FAILED
                try:
                    search_results = SearchResults(
                        job_id=job_id,
                        metric_names=METRICS.keys(),
                        n_peaks=self._ds.config['isotope_generation']['n_peaks'],
                        charge=self._ds.config['isotope_generation']['charge'],
                    )
                    img_store_type = self._ds.get_ion_img_storage_type(self._db)
                    sample_area_mask = make_sample_area_mask(imzml_parser.coordinates)
                    search_results.store(
                        moldb_ion_metrics_df,
                        moldb_ion_images_rdd,
                        sample_area_mask,
                        self._db,
                        self._img_store,
                        img_store_type,
                    )
                    job_status = JobStatus.FINISHED
                finally:
                    self._db.alter(
                        JOB_UPD_STATUS_FINISH,
                        params=(job_status, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), job_id),
                    )

    def _remove_annotation_jobs(self, moldbs):
        for moldb in moldbs:
            logger.info(
                f"Removing job results ds_id: {self._ds.id}, ds_name: {self._ds.name}, "
                f"db_name: {moldb.name}, db_version: {moldb.version}"
            )
            self._db.alter(
                'DELETE FROM job WHERE ds_id = %s and moldb_id = %s', params=(self._ds.id, moldb.id)
            )
            self._es.delete_ds(self._ds.id, moldb)

    def _moldb_ids(self):
        completed_moldb_ids = {
            db_id for (_, db_id) in self._db.select(JOB_ID_MOLDB_ID_SEL, params=(self._ds.id,))
        }
        new_moldb_ids = set(self._ds.config['database_ids'])
        return completed_moldb_ids, new_moldb_ids

    def _save_data_from_raw_ms_file(self, imzml_parser):
        ms_file_path = imzml_parser.filename
        ms_file_type_config = SMConfig.get_ms_file_handler(ms_file_path)
        dims = get_ds_dims(imzml_parser.coordinates)
        acq_geometry = make_acq_geometry(
            ms_file_type_config['type'], ms_file_path, self._ds.metadata, dims
        )
        self._ds.save_acq_geometry(self._db, acq_geometry)
        self._ds.save_ion_img_storage_type(self._db, ms_file_type_config['img_storage_type'])

    def _copy_input_data(self, ds):
        logger.info('Copying input data')
        self._ds_data_path = Path(self._sm_config['fs']['spark_data_path']) / ds.id
        if ds.input_path.startswith('s3a://'):
            self._ds_data_path.mkdir(parents=True, exist_ok=True)

            session = boto3.session.Session(
                aws_access_key_id=self._sm_config['aws']['aws_access_key_id'],
                aws_secret_access_key=self._sm_config['aws']['aws_secret_access_key'],
            )
            bucket_name, key = split_s3_path(ds.input_path)
            bucket = session.resource('s3').Bucket(bucket_name)  # pylint: disable=no-member
            for obj_sum in bucket.objects.filter(Prefix=key):
                local_file = str(self._ds_data_path / Path(obj_sum.key).name)
                logger.debug(f'Downloading s3a://{bucket_name}/{obj_sum.key} -> {local_file}')
                obj_sum.Object().download_file(local_file)
        else:
            rmtree(self._ds_data_path, ignore_errors=True)
            copytree(src=ds.input_path, dst=self._ds_data_path)

    def cleanup(self):
        if self._sc:
            self._sc.stop()
        logger.debug(f'Cleaning dataset temp dir {self._ds_data_path}')
        rmtree(self._ds_data_path, ignore_errors=True)

    def run(self, ds):
        """Starts dataset annotation job.

        Annotation job consists of several steps:
            * Copy input data to the engine work dir
            * Generate and save to the database theoretical peaks
              for all formulas from the molecule database
            * Molecules search. The most compute intensive part
              that uses most the cluster resources
            * Computing FDR per molecular database and filtering the results
            * Saving the results: metrics saved in the database, images in the Image service

        Args:
            ds (sm.engine.dataset.Dataset): dataset to annotate
        """
        try:
            logger.info('*' * 150)
            start = time.time()

            self._es = ESExporter(self._db, self._sm_config)
            self._ds = ds

            if self._sm_config['rabbitmq']:
                self._status_queue = QueuePublisher(
                    config=self._sm_config['rabbitmq'], qdesc=SM_DS_STATUS, logger=logger
                )
            else:
                self._status_queue = None

            logger.info('_configure_spark spark')
            self._configure_spark()
            logger.info('_copy_input_data')
            self._copy_input_data(ds)
            logger.info('create_imzml_parser')
            imzml_parser = self.create_imzml_parser()
            # self._save_data_from_raw_ms_file(imzml_parser)
            self._img_store.storage_type = 'fs'

            logger.info(f'Dataset config:\n{pformat(self._ds.config)}')

            completed_moldb_ids, new_moldb_ids = self._moldb_ids()
            # self._remove_annotation_jobs(
            #     molecular_db.find_by_ids(completed_moldb_ids - new_moldb_ids)
            # )
            self._run_annotation_jobs(
                imzml_parser, molecular_db.find_by_ids(new_moldb_ids - completed_moldb_ids)
            )

            logger.info("All done!")
            minutes, seconds = divmod(int(round(time.time() - start)), 60)
            logger.info(f'Time spent: {minutes} min {seconds} sec')
        finally:
            self.cleanup()
            logger.info('*' * 150)
