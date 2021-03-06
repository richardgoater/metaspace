
import gql from 'graphql-tag'
import reportError from '../../lib/reportError'

import { restoreImageViewerState } from './state'
import { restoreIonImageState } from './ionImageState'

export default async($apollo: any, id: string, datasetId: string) => {
  try {
    const result: any = await $apollo.query({
      query: gql`query fetchImageViewerSnapshot($id: String!, $datasetId: String!) {
        imageViewerSnapshot(id: $id, datasetId: $datasetId) {
          version
          snapshot
          annotations {
            id
            ion
            mz
            isotopeImages {
              minIntensity,
              maxIntensity,
              url
            }
            possibleCompounds {
              name
            }
          }
        }
      }`,
      variables: {
        id,
        datasetId,
      },
    })

    const { version, snapshot, annotations } = result.data.imageViewerSnapshot
    const parsed = JSON.parse(snapshot)

    restoreImageViewerState({
      version,
      snapshot: parsed.imageViewer,
    })

    restoreIonImageState({
      version,
      snapshot: parsed.ionImage,
      annotations,
    })
  } catch (e) {
    reportError(e)
  }
}
