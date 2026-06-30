/**
 * 知识百科模块 — 图谱浏览 + 融合消歧管理
 */
import { API } from './config.js';

const ADMIN = `${API}/admin`;

// ── 初始化入口（由 filters.js switchView 调用） ──
window.kbInit = function() {
  kbRefreshStats();
  kbLoadNodes();
  kbLoadPending();
};

// ── 图谱统计 ──
window.kbRefreshStats = async function() {
  try {
    const r = await fetch(`${ADMIN}/graph/stats`);
    const d = await r.json();
    document.getElementById('kb-total-v').textContent = d.total_vertices;
    document.getElementById('kb-total-e').textContent = d.total_edges;
    document.getElementById('kb-equip').textContent = d.tags.equipment || 0;
    document.getElementById('kb-loc').textContent = d.tags.location || 0;
    document.getElementById('kb-event').textContent = d.tags.event || 0;
    document.getElementById('kb-person').textContent = d.tags.person || 0;
    document.getElementById('kb-org').textContent = d.tags.organization || 0;
    document.getElementById('kb-record').textContent = d.tags.datarecord || 0;
  } catch (e) {
    console.error('Stats failed:', e);
  }
};

// ── 图节点浏览 ──
let _nodePage = 0;
const NODE_PAGE_SIZE = 30;
let _nodeSearchTimer = null;

window.kbLoadNodes = function(page) {
  if (page !== undefined) _nodePage = page;
  clearTimeout(_nodeSearchTimer);
  _nodeSearchTimer = setTimeout(_doLoadNodes, 300);
};

async function _doLoadNodes() {
  const tag = document.getElementById('kb-node-type').value;
  const search = document.getElementById('kb-node-search').value;
  const grid = document.getElementById('kb-node-grid');

  grid.innerHTML = '<div style="padding:20px;color:#64748b">加载中...</div>';

  try {
    const params = new URLSearchParams({
      tag, search,
      limit: NODE_PAGE_SIZE,
      offset: _nodePage * NODE_PAGE_SIZE,
    });
    const r = await fetch(`${ADMIN}/graph/nodes?${params}`);
    const d = await r.json();

    if (!d.items.length) {
      grid.innerHTML = '<div style="padding:20px;color:#64748b">无匹配节点</div>';
    } else {
      grid.innerHTML = d.items.map(n => {
        const name = n.name || n.title || n.vid || '-';
        const subtitle = _getNodeSubtitle(n, tag);
        return `<div class="kb-node-card" onclick="kbShowNodeDetail('${n.vid}')">
          <div class="kb-node-name">${name}</div>
          <div class="kb-node-sub">${subtitle}</div>
          <div class="kb-node-vid">${n.vid}</div>
        </div>`;
      }).join('');
    }

    // 分页
    const pages = Math.ceil(d.total / NODE_PAGE_SIZE);
    const pag = document.getElementById('kb-node-pagination');
    if (pages <= 1) {
      pag.innerHTML = `<span style="color:#64748b">共 ${d.total} 个节点</span>`;
    } else {
      pag.innerHTML = `<span style="color:#64748b">共 ${d.total} 个，第 ${_nodePage + 1}/${pages} 页</span>
        <button ${_nodePage === 0 ? 'disabled' : ''} onclick="kbLoadNodes(${_nodePage - 1})">上一页</button>
        <button ${_nodePage >= pages - 1 ? 'disabled' : ''} onclick="kbLoadNodes(${_nodePage + 1})">下一页</button>`;
    }
  } catch (e) {
    grid.innerHTML = `<div style="padding:20px;color:#ef4444">加载失败: ${e.message}</div>`;
  }
}

function _getNodeSubtitle(n, tag) {
  if (tag === 'equipment') return [n.category, n.equipment_type, n.country].filter(Boolean).join(' · ');
  if (tag === 'location') return [n.location_type, n.region].filter(Boolean).join(' · ');
  if (tag === 'event') return [n.event_type, n.start_date].filter(Boolean).join(' · ');
  if (tag === 'person') return [n.role, n.country].filter(Boolean).join(' · ');
  if (tag === 'organization') return [n.org_type, n.country].filter(Boolean).join(' · ');
  return '';
}

// ── 节点详情 — 复用 entities.py 专用 API + detail.js 渲染 ──

const _vidTypeMap = {
  equip: 'equipment', person: 'person', loc: 'location',
  event: 'activity', org: 'organization', record: 'datarecord', dataset: 'dataset',
};

const _apiPathMap = {
  equipment: 'equipment', person: 'persons', location: 'locations',
  activity: 'activities', organization: 'persons', // org 复用 person API
};

window.kbShowNodeDetail = async function(vid) {
  // 从 VID 解析类型和 ID：equip_123 → type=equip, id=123
  const match = vid.match(/^(equip|person|loc|event|org|record|dataset)_(\d+)$/);
  if (!match) {
    alert('无法解析节点 VID: ' + vid);
    return;
  }

  const [, prefix, id] = match;
  const type = _vidTypeMap[prefix];
  const apiPath = _apiPathMap[type];
  if (!apiPath) {
    alert('暂不支持查看此类型: ' + type);
    return;
  }

  try {
    const resp = await fetch(`${API}/${apiPath}/${id}`);
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    // 复用 article-modal 弹窗，但用 detail.js 的渲染逻辑
    const modal = document.getElementById('article-modal');
    const bodyEl = modal.querySelector('.modal-body .content');

    // 简化的详情渲染（不依赖地图）
    let html = '';
    if (type === 'equipment') html = _renderKbEquipment(data);
    else if (type === 'person') html = _renderKbPerson(data);
    else if (type === 'location') html = _renderKbLocation(data);
    else if (type === 'activity') html = _renderKbActivity(data);

    modal.querySelector('h3').textContent = '节点详情';
    bodyEl.innerHTML = html;
    // 知识百科节点详情不需要 footer
    modal.querySelector('.modal-footer').style.display = 'none';
    modal.style.display = 'flex';

    // 绑定"在态势中查看"按钮
    const gotoBtn = bodyEl.querySelector('.kb-goto-dashboard');
    if (gotoBtn) {
      gotoBtn.onclick = () => {
        window.closeArticleModal();
        window.switchView('dashboard');
        // 等视图切换完成后打开实体详情
        setTimeout(() => {
          const dashType = type === 'activity' ? 'activity' : type;
          if (window.showDetail) window.showDetail(dashType, parseInt(id));
        }, 500);
      };
    }
  } catch (e) {
    alert('加载失败: ' + e.message);
  }
};

function _renderKbEquipment(data) {
  const eq = data.equipment || data.ship || {};
  const aliases = Array.isArray(eq.aliases) ? eq.aliases.join(', ') : (eq.aliases || '-');
  let html = `<div class="kb-detail-header"><h4>⚙️ ${eq.name || '-'}</h4>
    <div class="kb-detail-sub">${eq.designation || ''} · ${eq.category || ''}</div>
    <button class="kb-goto-dashboard" style="margin-top:8px;background:#1e293b;border:1px solid #334155;border-radius:4px;color:#60a5fa;font-size:11px;padding:4px 10px;cursor:pointer">🗺️ 在态势中查看</button></div>`;
  html += _detailSection('基础信息', [
    _detailRow('编号', eq.designation),
    _detailRow('类型', eq.equipment_type || eq.category),
    _detailRow('状态', eq.status || eq.state),
    _detailRow('母港', eq.home_base || eq.home_location),
    _detailRow('别名', aliases),
  ]);
  html += _renderKbRelations(data.relations);
  html += _renderKbArticles(data.articles);
  return html;
}

function _renderKbPerson(data) {
  const p = data.person || {};
  let html = `<div class="kb-detail-header"><h4>👤 ${p.name || '-'}</h4>
    <div class="kb-detail-sub">${p.rank || ''} · ${p.position || ''}</div>
    <button class="kb-goto-dashboard" style="margin-top:8px;background:#1e293b;border:1px solid #334155;border-radius:4px;color:#60a5fa;font-size:11px;padding:4px 10px;cursor:pointer">🗺️ 在态势中查看</button></div>`;
  html += _detailSection('基础信息', [
    _detailRow('军衔', p.rank),
    _detailRow('职务', p.position),
    _detailRow('军种', p.service_branch),
    _detailRow('机构', p.org_name),
    _detailRow('别名', Array.isArray(p.aliases) ? p.aliases.join(', ') : (p.aliases || '-')),
  ]);
  html += _renderKbRelations(data.relations);
  html += _renderKbArticles(data.articles);
  return html;
}

function _renderKbLocation(data) {
  const l = data.location || data || {};
  let html = `<div class="kb-detail-header"><h4>📍 ${l.name || '-'}</h4>
    <div class="kb-detail-sub">${l.location_type || ''} · ${l.country || ''}</div>
    <button class="kb-goto-dashboard" style="margin-top:8px;background:#1e293b;border:1px solid #334155;border-radius:4px;color:#60a5fa;font-size:11px;padding:4px 10px;cursor:pointer">🗺️ 在态势中查看</button></div>`;
  html += _detailSection('基础信息', [
    _detailRow('类型', l.location_type),
    _detailRow('国家', l.country),
    _detailRow('地区', l.region),
    _detailRow('坐标', l.lat ? `${Number(l.lat).toFixed(4)}°, ${Number(l.lng).toFixed(4)}°` : null),
  ]);
  if (l.description) html += `<div class="kb-detail-section"><div style="color:#94a3b8;font-size:12px">${l.description}</div></div>`;
  html += _renderKbRelations(data.relations);
  html += _renderKbArticles(data.articles);
  return html;
}

function _renderKbActivity(data) {
  const a = data.activity || data || {};
  let html = `<div class="kb-detail-header"><h4>🎯 ${a.name || '-'}</h4>
    <div class="kb-detail-sub">${a.activity_type || ''} · ${a.region || ''}</div>
    <button class="kb-goto-dashboard" style="margin-top:8px;background:#1e293b;border:1px solid #334155;border-radius:4px;color:#60a5fa;font-size:11px;padding:4px 10px;cursor:pointer">🗺️ 在态势中查看</button></div>`;
  html += _detailSection('基础信息', [
    _detailRow('类型', a.activity_type),
    _detailRow('开始', a.start_date),
    _detailRow('结束', a.end_date),
    _detailRow('区域', a.region),
  ]);
  if (a.description) html += `<div class="kb-detail-section"><div style="color:#94a3b8;font-size:12px">${a.description}</div></div>`;
  html += _renderKbRelations(data.relations);
  html += _renderKbArticles(data.articles);
  return html;
}

function _renderKbRelations(relations) {
  if (!relations?.length) return '';
  return `<div class="kb-detail-section"><h5>关联实体 (${relations.length})</h5>
    ${relations.map(r => `<div class="kb-edge-item">
      <span class="kb-edge-type">${r.relation || r.direction || ''}</span>
      ${r.direction === 'incoming' ? '←' : '→'}
      <span class="kb-edge-target" onclick="kbShowNodeDetail('${(r.related_type || '').replace('equipment','equip').replace('person','person').replace('location','loc').replace('activity','event').replace('organization','org')}_${r.related_id}')">${r.related_name || r.related_id}</span>
      <span style="color:#475569;font-size:11px">(${r.related_type || ''})</span>
    </div>`).join('')}
  </div>`;
}

function _renderKbArticles(articles) {
  if (!articles?.length) return '';
  return `<div class="kb-detail-section"><h5>相关报道 (${articles.length})</h5>
    ${articles.slice(0, 5).map(a => {
      const date = a.published_at ? new Date(a.published_at).toLocaleDateString('zh-CN') : '';
      return `<div class="kb-article-card">
        <div class="kb-article-title" onclick="window.showArticleDetail(${a.id})">
          <span style="color:#e0e6ed;font-size:12px">${a.title}</span>
          <span style="color:#475569;font-size:11px;margin-left:4px;white-space:nowrap">${date}</span>
          <span style="color:#60a5fa;font-size:10px;margin-left:auto">详情 ▶</span>
        </div>
      </div>`;
    }).join('')}
  </div>`;
}

function _detailSection(title, rows) {
  return `<div class="kb-detail-section"><h5>${title}</h5>
    <div class="kb-detail-table">${rows.filter(r => r).join('')}</div></div>`;
}

function _detailRow(label, value) {
  if (!value || value === '-') return '';
  return `<div class="kb-detail-row"><span class="kb-detail-label">${label}</span><span class="kb-detail-value">${value}</span></div>`;
}

