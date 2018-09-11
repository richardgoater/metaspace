import { mount } from '@vue/test-utils';
import MetaspaceHeader from './MetaspaceHeader.vue';
import router from '../../router';
import { initMockGraphqlClient, provide } from '../../../tests/utils/mockGraphqlClient';
import store from '../../store/index';
import Vue from 'vue';
import Vuex from 'vuex';
import { sync } from 'vuex-router-sync';

Vue.use(Vuex);
sync(store, router);

describe('MetaspaceHeader', () => {
  it('should match snapshot (logged out)', async () => {
    initMockGraphqlClient({
      Query: () => ({
        currentUser: () => null // Prevent automatic mocking
      })
    });
    const wrapper = mount(MetaspaceHeader, { store, router, provide, sync: false });
    await Vue.nextTick();

    expect(wrapper).toMatchSnapshot();
  });

  it('should match snapshot (logged in)', async () => {
    initMockGraphqlClient({
      Query: () => ({
        currentUser: () => ({
          id: '123',
          name: 'Test User',
          email: 'test@example.com',
          primaryGroup: {
            group: {
              id: '456',
              name: 'Test Group'
            }
          }
        })
      })
    });
    const wrapper = mount(MetaspaceHeader, { store, router, provide, sync: false });
    await Vue.nextTick();

    expect(wrapper).toMatchSnapshot();
  });

  it('should include current filters in annotations & dataset links', async () => {
    router.push({path: '/annotations', query: {db: 'HMDB', organism: 'human'}});
    const wrapper = mount(MetaspaceHeader, { store, router, provide, sync: false });
    await Vue.nextTick();

    expect(wrapper.find('#annotations-link').attributes().href).toEqual("#/annotations?db=HMDB&organism=human");
    expect(wrapper.find('#datasets-link').attributes().href).toEqual("#/datasets?organism=human"); // db isn't a valid filter for datasets
  });
});