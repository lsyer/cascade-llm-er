/**
 * 导航 + 实体筛选器
 */
import { state } from './state.js';
import { renderAll } from './render.js';
import { loadPersons, loadArticles } from './data.js';
import { API, EQUIP_COLORS, TYPE_LABELS } from './config.js';

let currentType = 'equipment';
let currentSubFilter = null;
let _articleStatusFilter = null;
let _articlePage = 0;
const PAGE_SIZE = 50;

export { currentType, currentSubFilter };

// ── 视图切换 ──

window.switchView = async function(view) {
  document.querySelectorAll('.nav-link').forEach(n => n.classList.toggle('active', n.dataset.view === view));

  const appEl = document.getElementById('app');
  const toolbar = document.querySelector('.toolbar');
  const legend = document.querySelector('.legend');
  const pageArticles = document.getElementById('page-articles');
  const pageKnowledge = document.getElementById('page-knowledge');
  const pagePending = document.getElementById('page-pending');
  const pageResearch = document.getElementById('page-research');

  pageKnowledge.style.display = 'none';
  pagePending.style.display = 'none';
  if (pageResearch) pageResearch.style.display = 'none';
  if (window.chatHide) window.chatHide();

  if (view === 'dashboard') {
    appEl.style.display = '';
    if (toolbar) toolbar.style.display = '';
    if (legend) legend.style.display = '';
    pageArticles.style.display = 'none';
    // 延迟初始化地图（避免 WebGL 错误破坏模块加载链）
    if (window.initMap) window.initMap();
    renderSubFilters();
    renderAll();
  } else if (view === 'knowledge') {
    appEl.style.display = 'none';
    if (toolbar) toolbar.style.display = 'none';
    if (legend) legend.style.display = 'none';
    pageArticles.style.display = 'none';
    pageKnowledge.style.display = '';
    if (window.kbInit) window.kbInit();
  } else if (view === 'pending') {
    appEl.style.display = 'none';
    if (toolbar) toolbar.style.display = 'none';
    if (legend) legend.style.display = 'none';
    pageArticles.style.display = 'none';
    pagePending.style.display = '';
    if (window.kbLoadPending) window.kbLoadPending();
  } else if (view === 'research') {
    appEl.style.display = 'none';
    if (toolbar) toolbar.style.display = 'none';
    if (legend) legend.style.display = 'none';
    pageArticles.style.display = 'none';
    pageKnowledge.style.display = 'none';
    pagePending.style.display = 'none';
    pageResearch.style.display = '';
    if (window.chatHide) window.chatHide();
    if (window.researchInit) window.researchInit();
  } else if (view === 'chat') {
    appEl.style.display = 'none';
    if (toolbar) toolbar.style.display = 'none';
    if (legend) legend.style.display = 'none';
    pageArticles.style.display = 'none';
    pageKnowledge.style.display = 'none';
    pagePending.style.display = 'none';
    pageResearch.style.display = 'none';
    if (window.chatShow) window.chatShow();
  } else {
    appEl.style.display = 'none';
    if (toolbar) toolbar.style.display = 'none';
    if (legend) legend.style.display = 'none';
    pageArticles.style.display = '';
    _articlePage = 0;
    await refreshArticlesTable();
  }
};

// ── 实体 Tab ──

export function initTabs() {
  document.querySelectorAll('.entity-tab').forEach(tab => {
    tab.addEventListener('click', async () => {
      document.querySelectorAll('.entity-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentType = tab.dataset.type;
      currentSubFilter = null;
      if (currentType === 'person') await loadPersons();
      renderSubFilters();
      renderAll();
    });
  });

  // 导航栏事件
  document.querySelectorAll('.nav-link').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });

  renderSubFilters();
}

function renderSubFilters() {
  const container = document.getElementById('sub-filters');
  let types = [];
  if (currentType === 'equipment') types = [...new Set(state.allData.equipment.map(e => e.category).filter(Boolean))];
  else if (currentType === 'location') types = [...new Set(state.allData.locations.map(l => l.location_type).filter(Boolean))];
  else if (currentType === 'activity') types = [...new Set(state.allData.activities.map(a => a.activity_type).filter(Boolean))];
  if (!types.length) { container.innerHTML = ''; return; }
  const all = `<button class="sub-filter${!currentSubFilter ? ' active' : ''}" data-type="">全部</button>`;
  container.innerHTML = all + types.map(t => `<button class="sub-filter${currentSubFilter === t ? ' active' : ''}" data-type="${t}">${t}</button>`).join('');
  container.querySelectorAll('.sub-filter').forEach(btn => {
    btn.addEventListener('click', () => {
      currentSubFilter = btn.dataset.type || null;
      container.querySelectorAll('.sub-filter').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderAll();
    });
  });
}

// ── 原始数据表格 ──

async function refreshArticlesTable() {
  const data = await loadArticles(_articlePage * PAGE_SIZE, _articleStatusFilter);
  renderArticleFilters(data.total);
  renderArticleTable(data.items);
  renderArticlePagination(data.total);
}

function renderArticleFilters(total) {
  const container = document.getElementById('article-filters');
  const filters = [
    { key: '', label: '全部' },
    { key: 'done', label: '✅ 已抽取' },
    { key: 'failed', label: '❌ 失败' },
    { key: 'pending', label: '⏳ 待处理' },
  ];
  container.innerHTML = `<span style="color:#64748b;font-size:12px;padding:2px 6px">共 ${total} 篇</span>` +
    filters.map(f =>
      `<button class="sub-filter${_articleStatusFilter === f.key || (!_articleStatusFilter && !f.key) ? ' active' : ''}" data-status="${f.key}" style="font-size:11px">${f.label}</button>`
    ).join('');
  container.querySelectorAll('.sub-filter').forEach(btn => {
    btn.addEventListener('click', async () => {
      container.querySelectorAll('.sub-filter').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      _articleStatusFilter = btn.dataset.status || null;
      _articlePage = 0;
      await refreshArticlesTable();
    });
  });
}

function renderArticleTable(items) {
  const tbody = document.getElementById('articles-tbody');
  if (!items.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#64748b;padding:40px">暂无文章</td></tr>';
    return;
  }
  tbody.innerHTML = items.map(a => {
    const date = a.published_at ? new Date(a.published_at).toLocaleDateString('zh-CN') : '-';
    const sc = a.status || 'pending';
    const statusLabels = { done: '✅ 已抽取', pending: '⏳ 待处理', processing: '🔄 处理中', failed: '❌ 失败' };
    // 操作按钮
    let actions = `<button class="at-btn view" onclick="showArticleDetail(${a.id})">查看</button>`;
    const noContent = !a.content_len || a.content_len === 0;
    if (noContent) {
      actions += `<button class="at-btn refetch" onclick="refetchArticleRow(${a.id}, this)">📥 采集</button>`;
    }
    if (sc === 'failed' || sc === 'pending') {
      actions += `<button class="at-btn retry" onclick="retryArticleRow(${a.id}, this)">▶ 重试</button>`;
    } else if (sc === 'done' && (a.entity_count || 0) > 0) {
      actions += `<button class="at-btn reprocess" onclick="reprocessArticleRow(${a.id}, this)">🔄 撤回重做</button>`;
    }
    return `<tr>
      <td style="color:#64748b">${a.id}</td>
      <td><span class="at-title" onclick="showArticleDetail(${a.id})" title="${a.title}">${a.title}</span></td>
      <td><span class="at-status ${sc}">${statusLabels[sc] || sc}</span></td>
      <td style="color:#94a3b8">${a.entity_count || 0}</td>
      <td style="color:#64748b">${date}</td>
      <td style="color:#64748b">${a.content_len || 0}</td>
      <td>${actions}</td>
    </tr>`;
  }).join('');
}

function renderArticlePagination(total) {
  const pages = Math.ceil(total / PAGE_SIZE);
  const el = document.getElementById('articles-pagination');
  if (pages <= 1) { el.innerHTML = `<span>共 ${total} 篇</span>`; return; }
  let html = `<span>共 ${total} 篇，第 ${_articlePage + 1}/${pages} 页</span>`;
  html += `<button ${_articlePage === 0 ? 'disabled' : ''} onclick="articlePage(-1)">上一页</button>`;
  html += `<button ${_articlePage >= pages - 1 ? 'disabled' : ''} onclick="articlePage(1)">下一页</button>`;
  el.innerHTML = html;
}

window.articlePage = async function(delta) {
  _articlePage += delta;
  if (_articlePage < 0) _articlePage = 0;
  await refreshArticlesTable();
};

// ── 文章操作 ──

window.retryArticleRow = async function(articleId, btn) {
  btn.textContent = '⏳'; btn.disabled = true;
  try {
    await fetch(`${API}/admin/articles/${articleId}/reset`, { method: 'POST' });
    await fetch(`${API}/admin/extract`, { method: 'POST' });
    btn.textContent = '✅'; btn.className = 'at-btn view';
    setTimeout(() => refreshArticlesTable(), 15000);
  } catch { btn.textContent = '❌'; }
};

window.refetchArticleRow = async function(articleId, btn) {
  btn.textContent = '⏳'; btn.disabled = true;
  try {
    const resp = await fetch(`${API}/admin/articles/${articleId}/refetch`, { method: 'POST' });
    const data = await resp.json();
    if (data.status === 'ok') {
      btn.textContent = `✅ ${data.new_len}字`;
      setTimeout(() => refreshArticlesTable(), 2000);
    } else {
      btn.textContent = '❌';
      alert(data.message || '采集失败');
    }
  } catch { btn.textContent = '❌'; }
};

window.reprocessArticleRow = async function(articleId, btn) {
  if (!confirm('撤回并重新处理？将删除孤立实体后重新抽取。')) return;
  btn.textContent = '⏳'; btn.disabled = true;
  try {
    await fetch(`${API}/admin/reprocess/${articleId}`, { method: 'POST' });
    btn.textContent = '✅';
    setTimeout(() => refreshArticlesTable(), 15000);
  } catch { btn.textContent = '❌'; }
};

window.retryAllFailed = async function() {
  const btn = document.getElementById('btn-retry-all');
  btn.textContent = '⏳ 重试中...'; btn.disabled = true;
  try {
    // 获取所有 failed 文章并重置
    const data = await loadArticles(0, 'failed');
    for (const a of data.items) {
      if (a.status === 'failed') {
        await fetch(`${API}/admin/articles/${a.id}/reset`, { method: 'POST' });
      }
    }
    await fetch(`${API}/admin/extract`, { method: 'POST' });
    btn.textContent = '✅ 已提交';
    setTimeout(() => { btn.textContent = '▶ 重试全部失败'; btn.disabled = false; refreshArticlesTable(); }, 20000);
  } catch { btn.textContent = '❌ 失败'; btn.disabled = false; }
};

window.triggerNewScrape = async function() {
  const btn = event?.target;
  if (btn) { btn.textContent = '📡 采集中...'; btn.disabled = true; }
  try {
    await fetch(`${API}/admin/scrape`, { method: 'POST' });
    if (btn) btn.textContent = '✅ 采集已启动';
    setTimeout(() => { if (btn) { btn.textContent = '📡 采集新文章'; btn.disabled = false; } refreshArticlesTable(); }, 30000);
  } catch { if (btn) { btn.textContent = '📡 采集新文章'; btn.disabled = false; } }
};
