<template>
  <div>
    <div class="page-header">
      <h2><el-icon><DataAnalysis /></el-icon> 因子库</h2>
      <p>浏览数据库中已保存的因子公式，按版本和集合筛选</p>
    </div>

    <!-- Filters -->
    <el-card shadow="hover" style="margin-bottom:16px;">
      <el-form inline size="default">
        <el-form-item label="集合">
          <el-select v-model="selectedCollection" clearable placeholder="全部集合" style="width:220px" @change="fetchFactors">
            <el-option v-for="c in collections" :key="c" :label="c" :value="c" />
          </el-select>
        </el-form-item>
        <el-form-item label="版本">
          <el-select v-model="selectedVersion" filterable clearable placeholder="全部版本" style="width:280px" @change="fetchFactors">
            <el-option v-for="v in filteredVersions" :key="v" :label="v" :value="v" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="fetchFactors" :loading="loading">
            <el-icon><Refresh /></el-icon> 刷新
          </el-button>
        </el-form-item>
        <el-form-item>
          <el-tag type="info">共 {{ factors.length }} 条记录</el-tag>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Factor Table -->
    <el-card shadow="hover">
      <el-table
        :data="factors"
        stripe
        border
        size="small"
        max-height="600"
        style="width:100%"
        @selection-change="handleSelectionChange"
      >
        <el-table-column type="selection" width="45" />
        <el-table-column prop="collection" label="集合" width="160" sortable show-overflow-tooltip />
        <el-table-column prop="version" label="版本" width="200" sortable show-overflow-tooltip />
        <el-table-column prop="factor_name" label="因子名称" width="140" sortable />
        <el-table-column prop="formula" label="公式" min-width="400" show-overflow-tooltip>
          <template #default="{ row }">
            <el-tooltip :content="row.formula" placement="top-start" :show-after="300">
              <code style="font-size:12px; color:#606266;">{{ row.formula }}</code>
            </el-tooltip>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="170" sortable show-overflow-tooltip />
      </el-table>

      <!-- Batch actions -->
      <div style="margin-top:12px; display:flex; align-items:center; gap:12px;" v-if="selectedFactors.length">
        <el-button type="success" @click="goBacktest">
          <el-icon><TrendCharts /></el-icon> 对选中因子({{ selectedFactors.length }})进行回测
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { getVersions, getFactors } from '../api'

const router = useRouter()
const loading = ref(false)
const collections = ref([])
const versionMap = ref({})
const allVersions = ref([])
const selectedCollection = ref('')
const selectedVersion = ref('')
const factors = ref([])
const selectedFactors = ref([])

const filteredVersions = computed(() => {
  if (selectedCollection.value && versionMap.value[selectedCollection.value]) {
    return versionMap.value[selectedCollection.value]
  }
  return allVersions.value
})

const fetchVersions = async () => {
  try {
    const { data } = await getVersions()
    collections.value = data.collections || []
    versionMap.value = data.version_map || {}
    allVersions.value = data.all_versions || []
  } catch (err) {
    ElMessage.error('获取版本列表失败: ' + err.message)
  }
}

const fetchFactors = async () => {
  loading.value = true
  try {
    const params = {}
    if (selectedVersion.value) params.version = selectedVersion.value
    if (selectedCollection.value) params.collection = selectedCollection.value
    const { data } = await getFactors(params)
    factors.value = data.factors || []
  } catch (err) {
    ElMessage.error('获取因子列表失败: ' + err.message)
  } finally {
    loading.value = false
  }
}

const handleSelectionChange = (rows) => {
  selectedFactors.value = rows
}

const goBacktest = () => {
  if (!selectedFactors.value.length) return
  // Group by version + collection
  const first = selectedFactors.value[0]
  const fcNames = selectedFactors.value.map(f => f.factor_name)
  // Save to sessionStorage and navigate
  sessionStorage.setItem('backtest_prefill', JSON.stringify({
    version: first.version,
    collection: first.collection,
    fc_name_list: fcNames,
  }))
  router.push('/backtest')
}

onMounted(() => {
  fetchVersions()
  fetchFactors()
})
</script>

