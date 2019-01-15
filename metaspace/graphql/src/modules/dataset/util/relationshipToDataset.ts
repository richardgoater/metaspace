import {DatasetSource} from '../../../bindingTypes';
import {Context} from '../../../context';
import {Project as ProjectModel, UserProjectRoleOptions as UPRO} from '../../project/model';

interface DatasetRelationship {
  type: 'submitter' | 'group' | 'project';
  id: string;
  name: string;
}

/**
 * Returns a list of ways that the user is related to a dataset, sorted by importance
 * @param dataset
 * @param ctx
 */
export const relationshipToDataset = async (dataset: DatasetSource, ctx: Context): Promise<DatasetRelationship[]> => {
  const ds = dataset._source;
  const user = ctx.user;
  const relationships: DatasetRelationship[] = [];
  if (user != null) {
    if (user.id === ds.ds_submitter_id) {
      relationships.push({
        type: 'submitter',
        id: ds.ds_submitter_id,
        name: ds.ds_submitter_name || '',
      });
    }
    if (user.groupIds != null && ds.ds_group_id != null && user.groupIds.includes(ds.ds_group_id)) {
      relationships.push({
        type: 'group',
        id: ds.ds_group_id,
        name: ds.ds_group_name || '',
      });
    }
    if (ds.ds_project_ids != null && ds.ds_project_ids.length > 0) {
      const projectRoles = await user.getProjectRoles();
      const projectIds = ds.ds_project_ids
        .filter(id => projectRoles[id] === UPRO.MANAGER || projectRoles[id] === UPRO.MEMBER);
      if (projectIds.length > 0) {
        const projects = await ctx.entityManager.getRepository(ProjectModel).findByIds(projectIds);
        projects.forEach(({id, name}) => {
          relationships.push({ type: 'project', id, name })
        });
      }
    }
  }
  return relationships;
};