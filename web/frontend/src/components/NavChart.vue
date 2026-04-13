<template>
  <v-chart class="nav-chart" :option="chartOption" autoresize />
</template>

<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import {
  TitleComponent, TooltipComponent, LegendComponent,
  GridComponent, DataZoomComponent, ToolboxComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([
  LineChart, TitleComponent, TooltipComponent, LegendComponent,
  GridComponent, DataZoomComponent, ToolboxComponent, CanvasRenderer,
])

const props = defineProps({
  title: { type: String, default: '' },
  curveData: { type: Object, default: () => ({}) },
  showBaseline: { type: Boolean, default: true },
  height: { type: String, default: '400px' },
})

const chartOption = computed(() => {
  const d = props.curveData
  if (!d || !d.time || !d.time.length) {
    return { title: { text: props.title || '暂无数据', left: 'center' } }
  }

  const times = d.time.map(t => t.slice(0, 10))

  const series = [
    {
      name: '策略净值(Gross)',
      type: 'line',
      data: d.gross_nav,
      showSymbol: false,
      lineStyle: { width: 2 },
    },
    {
      name: '策略净值(Net)',
      type: 'line',
      data: d.net_nav,
      showSymbol: false,
      lineStyle: { width: 2 },
    },
  ]

  if (props.showBaseline && d.gross_nav_baseline_long) {
    series.push(
      {
        name: '基准多头',
        type: 'line',
        data: d.gross_nav_baseline_long,
        showSymbol: false,
        lineStyle: { width: 1, type: 'dashed' },
      },
      {
        name: '基准空头',
        type: 'line',
        data: d.gross_nav_baseline_short,
        showSymbol: false,
        lineStyle: { width: 1, type: 'dashed' },
      },
    )
  }

  return {
    title: { text: props.title, left: 'center', textStyle: { fontSize: 14 } },
    tooltip: {
      trigger: 'axis',
      formatter(params) {
        let s = `<b>${params[0].axisValueLabel}</b><br/>`
        params.forEach(p => {
          if (p.value != null) {
            s += `${p.marker} ${p.seriesName}: ${p.value.toFixed(4)}<br/>`
          }
        })
        return s
      },
    },
    legend: { top: 30, textStyle: { fontSize: 12 } },
    grid: { top: 70, left: 60, right: 30, bottom: 60 },
    xAxis: { type: 'category', data: times, axisLabel: { rotate: 30, fontSize: 10 } },
    yAxis: { type: 'value', axisLabel: { formatter: v => v.toFixed(2) } },
    dataZoom: [
      { type: 'inside', start: 0, end: 100 },
      { type: 'slider', start: 0, end: 100, height: 20, bottom: 10 },
    ],
    toolbox: {
      right: 20,
      feature: {
        saveAsImage: { title: '保存图片' },
        dataZoom: { title: { zoom: '缩放', back: '还原' } },
        restore: { title: '还原' },
      },
    },
    series,
  }
})
</script>

<style scoped>
.nav-chart {
  width: 100%;
  height: v-bind('props.height');
}
</style>


