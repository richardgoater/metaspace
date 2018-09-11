<template>
  <tag-filter :name="name" @destroy="destroy" :width="multiple ? 900 : 300">
    <el-select slot="edit"
               placeholder="Start typing name"
               remote filterable clearable
               :remote-method="fetchOptions"
               :loading="loading"
               loading-text="Loading matching entries..."
               no-match-text="No matches"
               :multiple="multiple"
               :multiple-limit=10
               :value="value"
               @change="onInput">
      <el-option v-for="(item, idx) in joinedOptions"
                 :label="item.label"
                 :value="item.value"
                 :key="item.value">
      </el-option>
    </el-select>

    <span slot="show" class="tf-value-span">
      <span v-if="valueAsArray.length === 1">
        {{ currentLabel }}
      </span>
      <span v-if="valueAsArray.length > 1" >
        ({{ value.length }} items selected)
      </span>
      <span v-if="valueAsArray.length === 0">
        (any)
      </span>
    </span>
  </tag-filter>
</template>

<script lang="ts">
  import TagFilter from './TagFilter.vue';
  import Vue from 'vue';
  import Component from 'vue-class-component';
  import { Prop, Watch } from 'vue-property-decorator';
  import searchableFilterQueries, { SearchableFilterKey, Option } from './searchableFilterQueries';


  @Component({
    components: {
      TagFilter
    }
  })
  export default class SearchableFilter extends Vue {

    @Prop({ type: String, required: true })
    name!: string;
    @Prop({ type: Boolean, default: false })
    multiple!: boolean;
    @Prop()
    value!: string[] | string | undefined;
    @Prop({ type: String, required: true })
    filterKey!: SearchableFilterKey;

    loading = false;
    options: Option[] = [];
    cachedOptions: Option[] = [];
    currentLabel = '';

    created() {
      this.fetchNames();
      this.fetchOptions('');
    }

    get valueAsArray(): string[] {
      if (this.multiple) {
        return this.value != null ? this.value as string[] : [];
      } else {
        return this.value != null && this.value !== '' ? [this.value as string] : [];
      }
    }

    get joinedOptions() {
      // adds/moves selected values to the top of the options list

      const valueToLabel: Record<string, string> = {};
      for (let {value, label} of this.cachedOptions)
        valueToLabel[value] = label;

      const values = this.valueAsArray.slice();
      const options = values.map(value => ({value, label: valueToLabel[value]}));

      // add currently selected values to the list
      for (let i = 0; i < this.options.length; i++) {
        const item = this.options[i];
        if (values.indexOf(item.value) == -1) {
          values.push(item.value);
          options.push(item);
        }
      }

      return options;
    }

    @Watch('value')
    async fetchNames() {
      const foundOptions: Option[] = [];
      const missingValues: string[] = [];
      this.valueAsArray.forEach(value => {
        const option = this.cachedOptions.find(option => option.value === value)
                    || this.options.find(option => option.value === value);
        if (option != null) {
          foundOptions.push(option)
        } else {
          missingValues.push(value);
        }
      });
      this.cachedOptions = foundOptions;

      if (missingValues.length > 0) {
        const options = await searchableFilterQueries[this.filterKey].getById(this.$apollo, this.valueAsArray);
        this.cachedOptions.push(...options);
      }

      if (this.valueAsArray.length === 1) {
        // data.options.length may be 0 if an invalid ID is passed due to URL truncation or a dataset becoming hidden
        this.currentLabel = foundOptions.length > 0 ? foundOptions[0].label : this.valueAsArray[0];
      }
    }

    async fetchOptions(query: string) {
      this.loading = true;

      try {
        this.options = await searchableFilterQueries[this.filterKey].search(this.$apollo, this.$store, query);
        this.loading = false;
      } catch (err) {
        this.options = [];
        throw err;
      }
    }

    onInput(val: string) {
      this.$emit('input', val);
      this.$emit('change', val);
    }

    destroy() {
      this.$emit('destroy');
    }
  };
</script>

<style>
  .el-select-dropdown.is-multiple .el-select-dropdown__wrap {
    max-height: 600px;
  }
</style>