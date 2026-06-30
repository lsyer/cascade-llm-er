/**
 * 全局状态 — 单一对象，所有模块共享引用
 */
export const state = {
  currentTab: 'equipment',
  currentSubFilter: null,
  allData: { equipment: [], persons: [], locations: [], activities: [] },
  overviewStats: {},
  markers: [],
  trackLines: [],
  heatmapLayer: null,
  _currentArticles: {},
};
