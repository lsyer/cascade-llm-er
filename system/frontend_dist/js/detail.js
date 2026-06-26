/**
 * 详情面板
 */
import { API, EQUIP_COLORS, TYPE_LABELS, LOC_LABELS, ACT_LABELS } from './config.js';
import { addTrackLine, clearTrackLines, flyTo, fitBoundsFromCoords } from './map.js';
import { state } from './state.js';

export async function showDetail(type, id) {
  const panel = document.getElementById('detail-panel');
  const content = document.getElementById('detail-content');
  try {
    const pathMap = { equipment: 'equipment', person: 'persons', location: 'locations', activity: 'activities' };
    const resp = await fetch(`${API}/${pathMap[type]}/${id}`);
    const data = await resp.json();

    state._currentArticles = {};
    (data.articles || []).forEach(a => { state._currentArticles[a.id] = a; });

    if (type === 'equipment') renderEquipmentDetail(data, content);
    else if (type === 'person') renderPersonDetail(data, content);
    else if (type === 'location') renderLocationDetail(data, content);
    else if (type === 'activity') renderActivityDetail(data, content);

    panel.classList.add('open');
  } catch (e) {
    content.innerHTML = `<div style="padding:20px;color:#ef4444">加载失败: ${e.message}</div>`;
    panel.classList.add('open');
  }
}
window.showDetail = showDetail;

export async function showArticleDetail(articleId) {
  const modal = document.getElementById('article-modal');
  const titleEl = modal.querySelector('.modal-header h3');
  const bodyEl = modal.querySelector('.modal-body .content');
  const footerEl = modal.querySelector('.modal-footer');

  titleEl.textContent = '加载中...';
  bodyEl.innerHTML = '';
  modal.style.display = 'flex';

  try {
    const resp = await fetch(`${API}/articles/${articleId}`);
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    const a = data.article;
    const entities = data.entities || [];
    const date = a.published_at ? new Date(a.published_at).toLocaleString('zh-CN') : '';
    const statusMap = {
      done: { label: '✅ 抽取成功', color: '#10b981' },
      pending: { label: '⏳ 待处理', color: '#f59e0b' },
      processing: { label: '🔄 处理中', color: '#3b82f6' },
      failed: { label: '❌ 抽取失败', color: '#ef4444' },
    };
    const st = statusMap[a.status] || { label: a.status, color: '#94a3b8' };
    const typeIcons = { equipment: '⚙️', person: '👤', location: '📍', event: '⚡', organization: '🏢', activity: '🎯' };

    const canRetry = (a.status === 'failed' || a.status === 'pending');
    const canReprocess = (a.status === 'done' && entities.length > 0);

    titleEl.textContent = a.title;

    bodyEl.innerHTML = `
      <div style="margin-bottom:12px;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
        <span style="color:${st.color};background:${st.color}15;padding:2px 8px;border-radius:3px;font-size:11px">${st.label}</span>
        <span style="color:#64748b;font-size:12px">${date}</span>
        <span style="color:#64748b;font-size:12px">${a.content.length}字</span>
        <span style="color:#64748b;font-size:12px">${entities.length}个实体</span>
        ${canRetry ? `<button onclick="retryArticleModal(${articleId})" style="background:#065f46;color:#6ee7b7;border:1px solid #10b981;border-radius:4px;padding:3px 10px;font-size:11px;cursor:pointer">▶ 重新抽取</button>` : ''}
        ${canReprocess ? `<button onclick="reprocessArticleModal(${articleId})" style="background:#374151;color:#d1d5db;border:1px solid #4b5563;border-radius:4px;padding:3px 10px;font-size:11px;cursor:pointer">🔄 撤回重做</button>` : ''}
      </div>
      <div style="font-size:13px;color:#94a3b8;line-height:1.8;white-space:pre-wrap;max-height:400px;overflow-y:auto;margin-bottom:12px">${a.content || '无内容'}</div>
      ${a.url ? `<div style="margin-bottom:12px"><a href="${a.url}" target="_blank" style="color:#60a5fa;font-size:12px;text-decoration:none">🔗 查看原文</a></div>` : ''}
      ${entities.length ? `
        <div style="border-top:1px solid #1e293b;padding-top:10px">
          <div style="font-size:12px;font-weight:600;color:#60a5fa;margin-bottom:8px">关联实体 (${entities.length})</div>
          ${entities.map(e => `
            <div class="modal-entity-item" onclick="goToEntity('${e.entity_type}',${e.entity_id})">
              <span class="eicon">${typeIcons[e.entity_type] || '●'}</span>
              <span class="ename">${e.entity_name}</span>
              <span class="etype">${e.entity_type}</span>
              ${e.confidence ? `<span class="econf">${(e.confidence * 100).toFixed(0)}%</span>` : ''}
            </div>
          `).join('')}
        </div>
      ` : `<div style="color:#64748b;font-size:12px">${a.status === 'done' ? '⚠️ 抽取完成但未提取到实体' : '尚未抽取'}</div>`}
    `;

    footerEl.style.display = 'none';
  } catch (e) {
    titleEl.textContent = '加载失败';
    bodyEl.innerHTML = `<div style="color:#ef4444">${e.message}</div>`;
  }
}
window.showArticleDetail = showArticleDetail;

window.retryArticleModal = async function(articleId) {
  const body = document.querySelector('#article-modal .modal-body .content');
  try {
    await fetch(`${API}/admin/articles/${articleId}/reset`, { method: 'POST' });
    await fetch(`${API}/admin/extract`, { method: 'POST' });
    body.innerHTML = '<div style="color:#3b82f6;padding:20px">🔄 已提交重新抽取，请关闭后稍后刷新查看结果</div>';
  } catch (e) { body.innerHTML = `<div style="color:#ef4444">重试失败: ${e.message}</div>`; }
};

window.reprocessArticleModal = async function(articleId) {
  if (!confirm('撤回并重新处理？')) return;
  const body = document.querySelector('#article-modal .modal-body .content');
  try {
    const r = await fetch(`${API}/admin/reprocess/${articleId}`, { method: 'POST' });
    const d = await r.json();
    body.innerHTML = `<div style="color:#10b981;padding:20px">✅ 已撤回：删除 ${d.deleted_entities?.length || 0} 个孤立实体，保留 ${d.kept_entities?.length || 0} 个共享实体。正在重新抽取...</div>`;
  } catch (e) { body.innerHTML = `<div style="color:#ef4444">撤回失败: ${e.message}</div>`; }
};

function renderEquipmentDetail(data, content) {
  const eq = data.equipment || data.ship;
  const cat = eq.category || '';
  const c = EQUIP_COLORS[cat] || '#94a3b8';
  const srcTag = data.source === 'seed'
    ? '<span style="background:#334155;padding:2px 8px;border-radius:3px;font-size:11px;color:#94a3b8;margin-left:6px">📋 预置</span>'
    : '<span style="background:#064e3b;padding:2px 8px;border-radius:3px;font-size:11px;color:#6ee7b7;margin-left:6px">📰 新闻</span>';

  content.innerHTML = `
    <div class="header"><h2 style="color:${c}">${eq.name}</h2>
    <div class="subtitle">${eq.designation || ''} · ${TYPE_LABELS[cat] || cat} ${srcTag}</div></div>
    ${section('基础信息', [
      row('编号', eq.designation || '-'),
      row('类型', TYPE_LABELS[cat] || cat),
      row('母港', `${eq.home_base || '-'}${eq.home_lat ? ` (${Number(eq.home_lat).toFixed(2)}°,${Number(eq.home_lng).toFixed(2)}°)` : ''}`),
      row('状态', eq.status || 'active'),
      row('别名', Array.isArray(eq.aliases) ? eq.aliases.join(', ') : (eq.aliases || '-')),
    ])}
    ${section('位置历史', (data.positions || []).slice(0, 15).map(p =>
      `<div class="timeline-item"><div class="date">${new Date(p.reported_at).toLocaleString('zh-CN')}</div><div class="event">📍 ${p.lat.toFixed(2)}°, ${p.lng.toFixed(2)}°${p.notes ? ' · ' + p.notes : ''}</div></div>`
    ).join('') || '<div style="color:#64748b;font-size:12px">暂无位置记录</div>', (data.positions || []).length)}
    ${data.relations?.length ? section('关联实体', data.relations.map(r =>
      `<div class="relation-item" onclick="showDetail('${r.related_type}',${r.related_id})">${r.direction === 'incoming' ? '← ' : ''}${r.relation}${r.direction === 'incoming' ? '' : ' →'} ${r.related_name || r.related_id} (${r.related_type})</div>`
    ).join('')) : ''}
    ${renderArticleList(data.articles)}
  `;

  clearTrackLines();
  if ((data.positions || []).length > 1) {
    const coords = data.positions.map(p => [p.lat, p.lng]);
    addTrackLine(coords, { color: c, weight: 2, opacity: 0.6, dash: [5, 5] });
    fitBoundsFromCoords(coords, [50, 50]);
  } else {
    const p0 = data.positions?.[0];
    const fl = p0?.lat || eq.home_lat, fg = p0?.lng || eq.home_lng;
    if (fl && fg) flyTo(fl, fg, 6);
  }
}

function renderPersonDetail(data, content) {
  const p = data.person;
  content.innerHTML = `
    <div class="header"><h2 style="color:#f472b6">👤 ${p.name}</h2>
    <div class="subtitle">${p.rank || ''} · ${p.position || ''}</div></div>
    ${section('基础信息', [row('军衔', p.rank || '-'), row('职务', p.position || '-'), row('军种', p.service_branch || '-')])}
    ${data.career?.length ? section('任职时间线', data.career.map(c =>
      `<div class="timeline-item"><div class="date">${c.reported_at ? new Date(c.reported_at).toLocaleDateString('zh-CN') : ''}</div><div class="event">${c.rank || ''}${c.rank && c.position ? ' · ' : ''}${c.position || ''}${c.service_branch ? ' (' + c.service_branch + ')' : ''}</div></div>`
    ).join('')) : ''}
    ${data.relations?.length ? section('关联实体', data.relations.map(r =>
      `<div class="relation-item" onclick="showDetail('${r.related_type}',${r.related_id})">${r.direction === 'incoming' ? '← ' : ''}${r.relation}${r.direction === 'incoming' ? '' : ' →'} ${r.related_name || r.related_id} (${r.related_type})</div>`
    ).join('')) : ''}
    ${renderArticleList(data.articles)}
  `;
}

function renderLocationDetail(data, content) {
  const l = data.location || data;
  const lt = l.location_type || '';
  content.innerHTML = `
    <div class="header"><h2 style="color:#eab308">📍 ${l.name}</h2>
    <div class="subtitle">${l.country || ''} · ${l.region || ''}</div></div>
    ${section('基础信息', [
      row('类型', LOC_LABELS[lt] || lt || '-'),
      row('国家', l.country || '-'),
      row('战区', l.region || '-'),
      row('坐标', l.lat ? `${l.lat.toFixed(4)}°, ${l.lng.toFixed(4)}°` : '-'),
    ])}
    ${l.description ? `<div class="detail-section"><div style="font-size:12px;color:#94a3b8">${l.description}</div></div>` : ''}
    ${data.relations?.length ? section('关联实体', data.relations.map(r =>
      `<div class="relation-item" onclick="showDetail('${r.related_type}',${r.related_id})">${r.direction === 'incoming' ? '← ' : ''}${r.relation}${r.direction === 'incoming' ? '' : ' →'} ${r.related_name || r.related_id} (${r.related_type})</div>`
    ).join('')) : ''}
    ${renderArticleList(data.articles)}
  `;
  if (l.lat && l.lng) flyTo(l.lat, l.lng, 8);
}

function renderActivityDetail(data, content) {
  const a = data.activity || data;
  const at = a.activity_type || '';
  content.innerHTML = `
    <div class="header"><h2 style="color:#f97316">⚡ ${a.name}</h2>
    <div class="subtitle">${ACT_LABELS[at] || at} · ${a.region || ''}</div></div>
    ${section('基础信息', [
      row('类型', ACT_LABELS[at] || '-'),
      row('开始', a.start_date || '-'),
      row('结束', a.end_date || '-'),
      row('区域', a.region || '-'),
    ])}
    ${a.description ? `<div class="detail-section"><div style="font-size:12px;color:#94a3b8">${a.description}</div></div>` : ''}
    ${data.relations?.length ? section('关联实体', data.relations.map(r =>
      `<div class="relation-item" onclick="showDetail('${r.related_type}',${r.related_id})">${r.direction === 'incoming' ? '← ' : ''}${r.relation}${r.direction === 'incoming' ? '' : ' →'} ${r.related_name || r.related_id} (${r.related_type})</div>`
    ).join('')) : ''}
    ${renderArticleList(data.articles)}
  `;
}

function section(title, html, count) {
  return `<div class="detail-section"><h3>${title}${count !== undefined ? ` (${count})` : ''}</h3>${html}</div>`;
}

function row(label, value) {
  return `<div class="detail-row"><span class="label">${label}</span><span class="value">${value}</span></div>`;
}

function renderArticleList(articles) {
  if (!articles?.length) return section('相关报道', '<div style="color:#64748b;font-size:12px">📋 尚无相关报道</div>', 0);
  const statusMap = { done: { label: '✅', color: '#10b981' }, pending: { label: '⏳', color: '#f59e0b' }, processing: { label: '🔄', color: '#3b82f6' }, failed: { label: '❌', color: '#ef4444' } };
  return section('相关报道', articles.slice(0, 10).map(a => {
    const st = statusMap[a.status] || { label: '', color: '#64748b' };
    const hasContent = a.content && a.content.length > 0;
    return `
    <div style="margin-bottom:8px;background:#0f172a;border:1px solid #1e293b;border-radius:6px;overflow:hidden">
      <div style="padding:8px 10px;cursor:pointer;display:flex;justify-content:space-between;align-items:center" onclick="this.parentElement.querySelector('.article-body').classList.toggle('open')">
        <div style="flex:1;min-width:0"><div style="color:#e0e6ed;font-size:12px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${a.title}</div><div style="font-size:10px;color:#64748b;margin-top:2px"><span style="color:${st.color}">${st.label}</span> ${a.published_at ? new Date(a.published_at).toLocaleDateString('zh-CN') : ''}</div></div>
        <span style="color:#64748b;font-size:11px;margin-left:6px">▼</span>
      </div>
      <div class="article-body"><div style="padding:0 10px 10px;border-top:1px solid #1e293b">
        <div style="font-size:12px;color:#94a3b8;line-height:1.6;margin-top:8px">${hasContent ? a.content.substring(0, 200) + (a.content.length > 200 ? '...' : '') : '无内容'}</div>
        <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap">
          <button onclick="showArticleDetail(${a.id})" style="background:#1e293b;border:1px solid #334155;border-radius:4px;color:#e0e6ed;font-size:11px;padding:3px 8px;cursor:pointer">📄 详情${hasContent ? ` (${a.content.length}字)` : ''}</button>
          <a href="${a.url}" target="_blank" style="color:#60a5fa;font-size:11px;text-decoration:none">🔗 原文</a>
        </div>
      </div></div>
    </div>`;
  }).join(''), articles.length);
}

export function closeDetail() {
  document.getElementById('detail-panel').classList.remove('open');
  clearTrackLines();
}
window.closeDetail = closeDetail;

// 关闭文章详情弹窗
window.closeArticleModal = function() {
  document.getElementById('article-modal').style.display = 'none';
};

// 从原始数据弹窗跳转到态势首页查看实体
window.goToEntity = function(entityType, entityId) {
  // 关闭弹窗
  document.getElementById('article-modal').style.display = 'none';
  // 切换到态势首页
  window.switchView('dashboard');
  // 等页面切换后打开实体详情
  setTimeout(() => {
    window.showDetail(entityType === 'event' ? 'activity' : entityType, entityId);
  }, 500);
};
