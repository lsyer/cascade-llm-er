/**
 * 渲染：地图标记 + 列表
 */
import { addMarker, clearMarkers, createCircleMarker } from './map.js';
import { state } from './state.js';
import { currentType, currentSubFilter } from './filters.js';
import { EQUIP_COLORS, TYPE_LABELS, LOC_LABELS, ACT_LABELS } from './config.js';

export function renderAll() {
  renderMapMarkers();
  renderEntityList();
}

function renderMapMarkers() {
  clearMarkers();

  if (currentType === 'equipment') {
    const items = currentSubFilter
      ? state.allData.equipment.filter(e => e.category === currentSubFilter)
      : state.allData.equipment;
    items.forEach(eq => {
      const lat = eq.lat || eq.home_lat;
      const lng = eq.lng || eq.home_lng;
      if (!lat || !lng) return;
      const cat = eq.category || '';
      const c = EQUIP_COLORS[cat] || '#94a3b8';
      const marker = createCircleMarker(lat, lng,
        { radius: 6, color: c, fillColor: c, fillOpacity: 0.7, weight: 1 },
        `<div class="popup-content"><h3>${eq.name}</h3><div class="info">${eq.designation || ''} · ${TYPE_LABELS[cat] || cat}<br/>${eq.home_base || ''}</div><a class="btn" onclick="showDetail('equipment',${eq.id})">详情 →</a></div>`
      );
      addMarker(marker);
    });
  } else if (currentType === 'location') {
    state.allData.locations.forEach(loc => {
      if (!loc.lat || !loc.lng) return;
      const lt = loc.location_type || '';
      const marker = createCircleMarker(loc.lat, loc.lng,
        { radius: 5, color: '#eab308', fillColor: '#eab308', fillOpacity: 0.7, weight: 2 },
        `<div class="popup-content"><h3>${loc.name}</h3><div class="info">${loc.country || ''} · ${LOC_LABELS[lt] || lt}</div><a class="btn" onclick="showDetail('location',${loc.id})">详情 →</a></div>`
      );
      addMarker(marker);
    });
  } else if (currentType === 'person') {
    // 人员一般没有坐标
  } else if (currentType === 'activity') {
    // 活动暂无地图标记
  }
}

function renderEntityList() {
  const list = document.getElementById('entity-list');

  if (currentType === 'equipment') {
    let items = state.allData.equipment;
    if (currentSubFilter) items = items.filter(e => e.category === currentSubFilter);
    list.innerHTML = items.map(eq => {
      const cat = eq.category || '', c = EQUIP_COLORS[cat] || '#94a3b8';
      const pos = eq.lat ? `📍 ${eq.lat.toFixed(1)}°, ${eq.lng.toFixed(1)}°` : `⚓ ${eq.home_base || '未知'}`;
      return `<div class="entity-item" onclick="showDetail('equipment',${eq.id})"><div class="name"><span style="color:${c}">●</span> ${eq.name}</div><div class="meta">${eq.designation || ''} · ${pos}</div><span class="tag" style="background:${c}20;color:${c}">${TYPE_LABELS[cat] || cat}</span></div>`;
    }).join('');
  } else if (currentType === 'person') {
    list.innerHTML = state.allData.persons.map(p => {
      return `<div class="entity-item" onclick="showDetail('person',${p.id})"><div class="name"><span style="color:#f472b6">●</span> ${p.name}</div><div class="meta">${p.rank || ''} · ${p.position || ''}</div><span class="tag" style="background:#f472b620;color:#f472b6">${p.service_branch || ''}</span></div>`;
    }).join('');
  } else if (currentType === 'location') {
    let items = state.allData.locations;
    if (currentSubFilter) items = items.filter(l => l.location_type === currentSubFilter);
    list.innerHTML = items.map(l => {
      const lt = l.location_type || '';
      return `<div class="entity-item" onclick="showDetail('location',${l.id})"><div class="name"><span style="color:#eab308">●</span> ${l.name}</div><div class="meta">${l.country || ''} · ${LOC_LABELS[lt] || lt}</div><span class="tag" style="background:#eab30820;color:#eab308">${l.region || ''}</span></div>`;
    }).join('');
  } else if (currentType === 'activity') {
    let items = state.allData.activities;
    if (currentSubFilter) items = items.filter(a => a.activity_type === currentSubFilter);
    list.innerHTML = items.map(a => {
      const at = a.activity_type || '';
      return `<div class="entity-item" onclick="showDetail('activity',${a.id})"><div class="name"><span style="color:#f97316">●</span> ${a.name}</div><div class="meta">${a.region || ''} · ${ACT_LABELS[at] || at}</div><span class="tag" style="background:#f9731620;color:#f97316">${a.start_date || ''}</span></div>`;
    }).join('');
  }
}

export function renderArticleList(articles) {
  const list = document.getElementById('entity-list');
  if (!articles || !articles.length) {
    list.innerHTML = '<div style="padding:10px;color:#64748b;font-size:12px">暂无文章</div>';
    return;
  }
  list.innerHTML = articles.map(a => {
    const statusColors = { done: '#10b981', pending: '#f59e0b', processing: '#3b82f6', failed: '#ef4444' };
    const statusLabels = { done: '✅ 已抽取', pending: '⏳ 待处理', processing: '🔄 处理中', failed: '❌ 失败' };
    const sc = statusColors[a.status] || '#94a3b8';
    const sl = statusLabels[a.status] || a.status;
    const date = a.published_at ? new Date(a.published_at).toLocaleDateString('zh-CN') : '';
    const entCount = a.entity_count || 0;
    // 失败/待处理的文章显示重试按钮
    const canRetry = (a.status === 'failed' || a.status === 'pending');
    const retryBtn = canRetry
      ? `<button onclick="event.stopPropagation();retryArticle(${a.id})" style="background:#374151;color:#f59e0b;border:1px solid #4b5563;border-radius:3px;padding:1px 6px;font-size:10px;cursor:pointer;margin-left:4px" title="重新抽取">▶</button>`
      : '';
    return `<div class="entity-item" onclick="showArticleDetail(${a.id})" style="cursor:pointer">
      <div class="name" style="font-size:12px;line-height:1.4">📰 ${a.title}</div>
      <div class="meta">${date}${a.content_len ? ` · ${a.content_len}字` : ''} · ${entCount}个实体</div>
      <span class="tag" style="background:${sc}20;color:${sc}">${sl}${retryBtn}</span>
    </div>`;
  }).join('');
}
