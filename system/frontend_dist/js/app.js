/**
 * 入口 — 初始化
 */
import { initTabs } from './filters.js';
import { initSearch, closeArticleModal } from './interactions.js';
import { loadData } from './data.js';
import { initMap } from './map.js';
import './knowledge.js';
import './pending.js';

// 将 initMap 暴露到全局，供 filters.js switchView 调用
window.initMap = initMap;

initTabs();
initSearch();
initMap();   // 初始化地图（默认页面就是 dashboard，不等 tab 切换）
loadData();
setInterval(loadData, 300000);

document.getElementById('article-modal').addEventListener('click', function (e) {
  if (e.target === this || e.target.classList.contains('modal-box')) closeArticleModal();
});
