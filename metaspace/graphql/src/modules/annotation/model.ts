import {
  Entity,
  PrimaryColumn,
  Column, ManyToOne, JoinColumn, OneToMany, JoinTable, Index, PrimaryGeneratedColumn,
} from 'typeorm';
import {MomentValueTransformer} from '../../utils/MomentValueTransformer';
import {Moment} from 'moment';

@Entity('coloc_job')
export class ColocJob {

  @PrimaryColumn({ type: 'uuid', default: () => 'uuid_generate_v1mc()' })
  id: string;

  @Column({ type: 'text', name: 'ds_id' })
  datasetId: string;

  @Column({ type: 'text', name: 'mol_db', nullable: true })
  molDb: string | null;

  @Column({ type: 'numeric', precision: 2, scale: 2 })
  fdr: number;

  @Column({ type: 'text' })
  algorithm: string;

  @Column({ type: 'timestamp without time zone', default: () => "(now() at time zone 'utc')",
    transformer: new MomentValueTransformer() })
  start: Moment;

  @Column({ type: 'timestamp without time zone', default: () => "(now() at time zone 'utc')",
    transformer: new MomentValueTransformer() })
  finish: Moment;

  @Column({ type: 'text', nullable: true })
  error: string | null;

  @Column({ type: 'int', array: true, name: 'sample_ion_ids' })
  sampleIonIds: number[];

  @OneToMany(type => ColocAnnotation, colocAnnotation => colocAnnotation.colocJob)
  @JoinTable({ name: 'dataset_project' })
  colocAnnotations: ColocAnnotation[];
}

@Entity('coloc_annotation')
export class ColocAnnotation {

  @PrimaryColumn({ type: 'uuid', name: 'coloc_job_id' })
  colocJobId: string;

  /** Index of the annotation's sfAdduct in the parent ColocJob's orderedMols array */
  @PrimaryColumn({ type: 'int', name: 'ion_id' })
  ionId: number;

  @Column({ type: 'int', array: true, name: 'coloc_ion_ids' })
  colocIonIds: number[];

  @Column({ type: 'float4', array: true, name: 'coloc_coeffs' })
  colocCoeffs: number[];

  @ManyToOne(type => ColocJob, {onDelete: 'CASCADE'})
  @JoinColumn({ name: 'coloc_job_id' })
  colocJob: ColocJob;
}

@Entity('ion')
export class Ion {
  @PrimaryGeneratedColumn({ type: 'int' })
  id: number;

  /** All components of the ion in one string, with charge as a '+'/'-' suffix e.g. "C21H41O6P-HO+HO3P-H2O-CO2+Na+" */
  @Index({ unique: true })
  @Column({ type: 'text' })
  ion: string;

  /** The formula of the molecule prior to any modifications, e.g. "C21H41O6P".
   *  Internally this has been called "sumFormula" or "sf", but this term is incorrect.
   *  It should be referred to as "formula" or "molecular formula". */
  @Column({ type: 'text' })
  formula: string;

  // /** Array of molecules lost, including the '-' prefix, e.g. ['-H2O','-CO2'] */
  // @Column({ type: 'text', name: 'neutral_losses' })
  // neutralLosses: string[];

  // /** Array of derivatizations, using '-' and '+' prefixes to indicate atoms lost/gained, e.g. ['-HO+HO3P'] */
  // @Column({ type: 'text' })
  // derivatizations: string[];

  @Column({ type: 'text' })
  adduct: string;

  /** 1 or -1 */
  @Column({ type: 'smallint' })
  charge: number;
}