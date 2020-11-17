import { Ref, computed } from '@vue/composition-api'

interface SliderProps {
  value: number
  min: number
  max: number
  step: number
}

interface Range {
  minX: number
  maxX: number
}

export type SliderThumbInstance = {
  x: Ref<number>,
  getValue: (x: number) => number,
  increment: (factor: number) => number,
  decrement: (factor: number) => number,
}

export default (getProps: () => SliderProps, range: Ref<Range>) : SliderThumbInstance => {
  return {
    getValue(x: number) {
      const { max, min, step } = getProps()

      const ratio = (x - range.value.minX) / (range.value.maxX - range.value.minX)
      let value = ratio * (max - min) + min

      if (step !== 0.0) {
        value = step * Math.floor((value / step))
      }

      return value
    },
    x: computed(() => {
      const { value, min, max } = getProps()
      const ratio = ((value - min) / (max - min))
      return Math.ceil(ratio * (range.value.maxX - range.value.minX) + range.value.minX)
    }),
    increment(factor) {
      const { value, step, min, max } = getProps()
      return Math.min(Math.max(value + (step * factor), min), max)
    },
    decrement(factor) {
      const { value, step, min, max } = getProps()
      return Math.min(Math.max(value - (step * factor), min), max)
    },
  }
}