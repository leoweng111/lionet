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
  titleTooltip: { type: String, default: '' },
  curveData: { type: Object, default: () => ({}) },
  showBaseline: { type: Boolean, default: true },
  height: { type: String, default: '400px' },
  splitDate: { type: String, default: '' },
  splitDates: { type: Array, default: () => [] },
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

  if (Array.isArray(d.extra_series)) {
    d.extra_series.forEach((s, idx) => {
      if (!s || !Array.isArray(s.data)) return
      series.push({
        name: s.name || `扩展序列${idx + 1}`,
        type: 'line',
        data: s.data,
        showSymbol: false,
        lineStyle: { width: 2, type: s.lineType || 'solid' },
      })
    })
  }

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

  // 处理分割日期：优先使用 splitDates 数组，否则使用 splitDate
  const splitDateList = props.splitDates.length ? props.splitDates : (props.splitDate ? [props.splitDate] : [])
  const splitMarkLines = []
  const splitColors = ['#f56c6c', '#e6a23c', '#67c23a']  // 红色、橙色、绿色，对应不同分界
  const splitLabels = ['样本外分界', '实际样本外分界', '分界3']  // 标签

  splitDateList.forEach((splitDate, idx) => {
    const rawSplitDate = String(splitDate || '').replace(/\//g, '-')
    const normalizedSplitDate = /^\d{8}$/.test(rawSplitDate)
      ? `${rawSplitDate.slice(0, 4)}-${rawSplitDate.slice(4, 6)}-${rawSplitDate.slice(6, 8)}`
      : rawSplitDate.slice(0, 10)
    const splitIdx = normalizedSplitDate
      ? times.findIndex(t => t >= normalizedSplitDate)
      : -1
    if (splitIdx >= 0) {
      splitMarkLines.push({
        xAxis: times[splitIdx],
        color: splitColors[idx] || splitColors[0],
        label: splitLabels[idx] || `分界${idx + 1}`,
      })
    }
  })

  if (splitMarkLines.length > 0 && series.length > 0) {
    const originalMarkLine = series[0].markLine || {}
    series[0].markLine = {
      ...originalMarkLine,
      symbol: 'none',
      lineStyle: { color: splitMarkLines[0].color, type: 'dashed', width: 2 },
      label: { show: true, formatter: splitMarkLines[0].label, color: splitMarkLines[0].color },
      data: splitMarkLines.map(m => ({ xAxis: m.xAxis })),
    }
    // 如果有多条分割线，可以通过额外的 markLine 系列显示不同颜色和标签，但 echarts 一个 series 的 markLine 只能有一种样式
    // 作为简化，暂时用第一条线的样式，所有线颜色相同
  }

  return {
    title: {
      text: props.title,
      left: 'center',
      tooltip: {
        show: Boolean(props.titleTooltip || props.title),
        formatter: props.titleTooltip || props.title,
      },
      textStyle: {
        fontSize: 14,
        width: 560,
        overflow: 'truncate',
        ellipsis: '...',
      },
    },
    tooltip: {
      trigger: 'axis',
      formatter(params) {
        let s = `<b>${params[0].axisValueLabel}</b><br/>`
        params.forEach(p => {
          if (p.value != null && p.value !== '' && Number.isFinite(Number(p.value))) {
            s += `${p.marker} ${p.seriesName}: ${Number(p.value).toFixed(4)}<br/>`
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


