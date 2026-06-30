/**
 * 交互：搜索、文章弹窗等
 */
import { API, EQUIP_COLORS, TYPE_LABELS } from './config.js';
import { state } from './state.js';
import { showDetail } from './detail.js';

export function initSearch() {
  const input = document.getElementById('search-input');
  let timer;
  input.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => doSearch(input.value), 300);
  });
}

async function doSearch(q) {
  if (q.length < 2) return;
  try {
    const resp = await fetch(`${API}/search?q=${encodeURIComponent(q)}`);
    const results = await resp.json();
    const list = document.getElementById('entity-list');
    if (!results.length) {
      list.innerHTML = '<div style="padding:10px;color:#64748b;font-size:12px">无匹配结果</div>';
      return;
    }
    list.innerHTML = results.map(r => {
      const colors = { equipment: '#3b82f6', person: '#f472b6', location: '#eab308', activity: '#f97316' };
      const c = colors[r.type] || '#94a3b8';
      return `<div class="entity-item" onclick="showDetail('${r.type}',${r.id})"><div class="name"><span style="color:${c}">●</span> ${r.name}</div><div class="meta">${r.subtitle || ''}</div><span class="tag" style="background:${c}20;color:${c}">${r.type}</span></div>`;
    }).join('');
  } catch (e) {
    console.error('Search failed:', e);
  }
}

export function closeArticleModal() {
  document.getElementById('article-modal').style.display = 'none';
}
window.closeArticleModal = closeArticleModal;

window.toggleSidebar = function() {
  const sidebar = document.getElementById('sidebar');
  const toggle = document.getElementById('sidebar-toggle');
  sidebar.classList.toggle('collapsed');
  document.body.classList.toggle('sidebar-collapsed');
  toggle.textContent = sidebar.classList.contains('collapsed') ? '◀' : '▶';
};

window.reprocessArticle = async function(articleId) {
  if (!confirm('确认撤回并重新处理？')) return;
  try {
    const resp = await fetch(`${API}/admin/reprocess/${articleId}`, { method: 'POST' });
    const result = await resp.json();
    if (result.deleted_entities) {
      alert(`已撤回: 删除 ${result.deleted_entities.length} 个孤立实体, 保留 ${result.kept_entities.length} 个共享实体\n正在重新抽取...`);
    }
    // Refresh detail after delay
    setTimeout(() => showArticleDetail(articleId), 5000);
  } catch (e) {
    alert('撤回失败: ' + e.message);
  }
};

window.openArticleModal = function(id) {
  const a = state._currentArticles[id];
  if (!a) return;
  const modal = document.getElementById('article-modal');
  modal.querySelector('h3').textContent = a.title;
  modal.querySelector('.content').innerHTML = a.content ? a.content.replace(/\n/g, '<br/>') : '无内容';
  modal.querySelector('.orig-link').href = a.url;
  modal.querySelector('.orig-link').textContent = a.url;
  const reprocessBtn = modal.querySelector('.reprocess-btn');
  reprocessBtn.onclick = async () => {
    reprocessBtn.textContent = '处理中...';
    reprocessBtn.disabled = true;
    try {
      const resp = await fetch(`${API}/admin/reprocess/${id}`, { method: 'POST' });
      const result = await resp.json();
      reprocessBtn.textContent = `✅ ${result.deleted_entities?.length || 0} 实体已删除`;
    } catch (e) {
      reprocessBtn.textContent = '❌ 失败';
    }
  };
  modal.style.display = 'flex';
};
