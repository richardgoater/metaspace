import { mount } from '@vue/test-utils'
import Vue from 'vue'
import Vuex from 'vuex'
import router from '../../../router'
import store from '../../../store'
import { sync } from 'vuex-router-sync'
import DatasetItem from './DatasetItem.vue'

Vue.use(Vuex)
sync(store, router)

describe('DatasetItem', () => {
  const user = { id: 'user' }
  const submitter = { id: 'submitter' }
  const admin = { id: 'admin', role: 'admin' }
  const dataset = {
    id: 'mockdataset',
    status: 'FINISHED',
    metadataJson: JSON.stringify({
      Metadata_Type: 'Imaging MS',
      MS_Analysis: {
        Detector_Resolving_Power: { mz: 1234, Resolving_Power: 123456 },
      },
    }),
    molDBs: ['HMDB-v2.5', 'HMDB-v4', 'CHEBI'],
    polarity: 'POSITIVE',
    fdrCounts: {
      dbName: 'HMDB-v2.5',
      levels: [10],
      counts: [20],
    },
    groupApproved: true,
    submitter,
    organism: 'organism',
    organismPart: 'organismPart',
    condition: 'condition',
    analyzer: {
      type: 'type'
    },
    projects: [],
    uploadDT: '2020-04-17T11:37:50.318667'
  }
  const unpublished = { name: 'project', publicationStatus: 'UNPUBLISHED' }
  const underReview = { name: 'project', publicationStatus: 'UNDER_REVIEW' }
  const published = { name: 'project', publicationStatus: 'PUBLISHED' }

  it('should match snapshot', () => {
    const propsData = {
      currentUser: submitter,
      dataset,
    }
    const wrapper = mount(DatasetItem, { router, store, propsData })
    expect(wrapper).toMatchSnapshot()
  })

  it('should be able to delete if unpublished', () => {
    const propsData = {
      currentUser: submitter,
      dataset: {
        ...dataset,
        projects: [unpublished]
      },
    }
    const wrapper = mount(DatasetItem, { router, store, propsData })
    expect(wrapper.find('.ds-delete').exists()).toBe(true)
  })

  it('should not show the publication status if cannot edit', () => {
    const propsData = {
      currentUser: user,
      dataset: {
        ...dataset,
        projects: [published]
      },
    }
    const wrapper = mount(DatasetItem, { router, store, propsData })
    expect(wrapper.find('.test-publication-status').exists()).toBe(false)
  })

  it('should not show the publication status if processing', () => {
    const propsData = {
      currentUser: submitter,
      dataset: {
        ...dataset,
        projects: [published],
        status: 'ANNOTATING'
      }
    }
    const wrapper = mount(DatasetItem, { router, store, propsData })
    expect(wrapper.find('.test-publication-status').exists()).toBe(false)
  })

  it('should show "Under review" status', () => {
    const propsData = {
      currentUser: submitter,
      dataset: {
        ...dataset,
        projects: [underReview]
      }
    }
    const wrapper = mount(DatasetItem, { router, store, propsData })
    expect(wrapper.find('.test-publication-status').text()).toBe('Under review')
  })

  it('should show "Published" status', () => {
    const propsData = {
      currentUser: submitter,
      dataset: {
        ...dataset,
        projects: [published]
      }
    }
    const wrapper = mount(DatasetItem, { router, store, propsData })
    expect(wrapper.find('.test-publication-status').text()).toBe('Published')
  })

  it('should prefer "Published" status', () => {
    const propsData = {
      currentUser: submitter,
      dataset: {
        ...dataset,
        projects: [published, underReview]
      }
    }
    const wrapper = mount(DatasetItem, { router, store, propsData })
    expect(wrapper.find('.test-publication-status').text()).toBe('Published')
  })

  it('should show admin options', () => {
    const propsData = {
      currentUser: admin,
      dataset: {
        ...dataset,
        projects: [published]
      }
    }
    const wrapper = mount(DatasetItem, { router, store, propsData })
    expect(wrapper.find('.ds-delete').exists()).toBe(true)
    expect(wrapper.find('.ds-reprocess').exists()).toBe(true)
  })
})