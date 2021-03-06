import gql from 'graphql-tag'

import { MolecularDB } from './moldb'

// Prefixing these with `Gql` because differently-cased variants are used elsewhere
export type GqlPolarity = 'POSITIVE' | 'NEGATIVE';
export type GqlJobStatus = 'QUEUED' | 'ANNOTATING' | 'FINISHED' | 'FAILED';

export interface DatasetDetailItem {
  id: string;
  name: string;
  submitter: {
    id: string | null;
    name: string;
  };
  principalInvestigator: {
    name: string;
    email: string | null;
  } | null;
  group: {
    id: string;
    name: string;
    shortName: string;
  };
  projects: {
    id: string;
    name: string;
    publicationStatus: String;
  }[],
  groupApproved: boolean;
  polarity: GqlPolarity;
  ionisationSource: string;
  analyzer: {
    type: string;
    resolvingPower: number;
  };
  organism: string | null;
  organismPart: string | null;
  condition: string | null;
  growthConditions: string | null;
  metadataJson: string;
  isPublic: boolean;
  molDBs: string[];
  databases: MolecularDB[];
  status: GqlJobStatus | null;
  metadataType: string;
  fdrCounts: {
    databaseId: number;
    dbName: string;
    dbVersion: string;
    levels: number[];
    counts: number[];
  };
  rawOpticalImageUrl: string;
  canDownload: boolean;
  uploadDT: string;
}

export const datasetDetailItemFragment =
  gql`fragment DatasetDetailItem on Dataset {
    id
    name
    submitter {
      id
      name
      email
    }
    principalInvestigator { name email }
    group { id name shortName }
    groupApproved
    projects { id name publicationStatus }
    polarity
    ionisationSource
    analyzer {
      type
      resolvingPower(mz: 400)
    }
    organism
    organismPart
    condition
    growthConditions
    metadataJson
    isPublic
    databases {
      id
      name
      version
    }
    status
    statusUpdateDT
    metadataType
    fdrCounts(inpFdrLvls: $inpFdrLvls, checkLvl: $checkLvl) {
      databaseId
      dbName
      dbVersion
      levels
      counts
    }
    thumbnailOpticalImageUrl
    ionThumbnailUrl
    canDownload
    uploadDT
  }`

export const datasetDetailItemsQuery =
  gql`query GetDatasets(
    $dFilter: DatasetFilter, $query: String, $inpFdrLvls: [Int!]!, $checkLvl: Int!, $offset: Int = 0
  ) {
    allDatasets(offset: $offset, limit: 100, filter: $dFilter, simpleQuery: $query) {
      ...DatasetDetailItem
    }
  }
  ${datasetDetailItemFragment}
  `

export const countDatasetsByStatusQuery =
  gql`query CountDatasets($dFilter: DatasetFilter, $query: String) {
    countDatasetsPerGroup(query: {filter: $dFilter, simpleQuery: $query, fields: [DF_STATUS]}) {
      counts {
        fieldValues
        count
      }
    }
  }`

export const countDatasetsQuery =
  gql`query CountDatasets($dFilter: DatasetFilter, $query: String) {
    countDatasets(filter: $dFilter, simpleQuery: $query)
  }`

export interface DatasetListItem {
  id: string;
  name: string;
  uploadDT: string;
}

export const datasetListItemsQuery =
  gql`query GetDatasets($dFilter: DatasetFilter, $query: String) {
    allDatasets(offset: 0, limit: 10000, filter: $dFilter, simpleQuery: $query) {
      id
      name
      uploadDT
    }
  }`

export type OpticalImageType = 'SCALED' | 'CLIPPED_TO_ION_IMAGE';

export interface OpticalImage {
  url: string;
  type: OpticalImageType;
  zoom: number;
  transform: number[][];
}
export const opticalImagesQuery =
  gql`query ($datasetId: String!, $type: OpticalImageType) {
    dataset(id: $datasetId) {
      id
      opticalImages(type: $type) {
        id
        url
        type
        zoom
        transform
      }
    }
  }`

export const rawOpticalImageQuery =
    gql`query Q($ds_id: String!) {
    rawOpticalImage(datasetId: $ds_id) {
      url
      transform
    }
  }`

export const createDatasetQuery =
  gql`mutation ($input: DatasetCreateInput!) {
      createDataset(input: $input, priority: 1)
  }`

export const deleteDatasetQuery =
  gql`mutation ($id: String!, $force: Boolean) {
    deleteDataset(id: $id, force: $force)
  }`

export const reprocessDatasetQuery =
  gql`mutation ($id: String!) {
    reprocessDataset(id: $id)
  }`

export const addOpticalImageQuery =
  gql`mutation ($imageUrl: String!,
                $datasetId: String!, $transform: [[Float]]!) {
    addOpticalImage(input: {datasetId: $datasetId,
                            imageUrl: $imageUrl, transform: $transform})
  }`

export const deleteOpticalImageQuery =
  gql`mutation ($id: String!) {
    deleteOpticalImage(datasetId: $id)
  }`

export const msAcqGeometryQuery =
  gql`query ($datasetId: String!) {
    dataset(id: $datasetId) {
      id
      acquisitionGeometry
    }
  }`

export const datasetVisibilityQuery =
  gql`query DatasetVisibility($id: String!) {
     datasetVisibility: dataset(id: $id) {
       id
       submitter { id name }
       group { id name }
       projects { id name }
     }
     currentUser { id }
   }`
export interface DatasetVisibilityQuery {
  datasetVisibility: DatasetVisibilityResult | null
  currentUser: { id: string } | null
}
export interface DatasetVisibilityResult {
  id: string;
  submitter: { id: string, name: string };
  group: { id: string, name: string } | null;
  projects: { id: string, name: string }[] | null;
}

export const datasetStatusUpdatedQuery = gql`subscription datasetStatusUpdated(
  $inpFdrLvls: [Int!] = [10],
  $checkLvl: Int = 10
) {
  datasetStatusUpdated {
    dataset {
      ...DatasetDetailItem
    }
    relationship {
      type
      id
      name
    }
    action
    stage
    isNew
  }
}
${datasetDetailItemFragment}`

export const datasetDeletedQuery = gql`subscription datasetDeleted {
  datasetDeleted {
    id
  }
}`

export const getDatasetDownloadLink = gql`query getDatasetDownloadLink ($datasetId: String!) {
  dataset(id: $datasetId) { downloadLinkJson }
}`
export interface GetDatasetDownloadLink {
  dataset: {
    downloadLinkJson: string | null
  }
}
export interface DownloadLinkJson {
  contributors: {name: string, institution?: string}[],
  license: {code: string, name: string, link: string},
  files: {filename: string, link: string}[],
}
