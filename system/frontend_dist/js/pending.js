/** 待决队列 — 系统级：抽取流程 + 全量扫描 → LLM 不确定时汇入此列表 */

import { API } from './config.js';

const ADMIN = `${API}/admin`;

let _pendingPage = 0;
const PENDING_PAGE_SIZE = 20;

window.kbLoadPending = async function(page) {
  if (page !== undefined) _pendingPage = page;
  const status = document.getElementById('kb-pending-filter').value;
  const tbody = document.getElementById('kb-pending-tbody');

  try {
    const params = new URLSearchParams({
      status: status || '',
      limit: PENDING_PAGE_SIZE,
      offset: _pendingPage * PENDING_PAGE_SIZE,
    });
    const r = await fetch(`${ADMIN}/pending-entities?${params}`);
    const d = await r.json();

    // 摘要
    document.getElementById('kb-pending-info').textContent =
      `共 ${d.total} 条${status ? '（筛选: ' + status + '）' : ''}`;

    if (!d.items.length) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#64748b;padding:20px">无数据</td></tr>';
    } else {
      tbody.innerHTML = d.items.map(item => {
        const sim = (item.similarity * 100).toFixed(0) + '%';
        const simClass = item.similarity >= 0.85 ? 'kb-sim-high' : item.similarity >= 0.5 ? 'kb-sim-mid' : 'kb-sim-low';
        const verdictMap = { merge: '融合', disambiguate: '消歧', reject: '拒绝', new: '新建', auto_merge: '自动融合', l2_uncertain: 'L2不确定', unclear: 'LLM不确定' };
        const verdict = item.llm_verdict ? (verdictMap[item.llm_verdict] || item.llm_verdict) : '-';
        const statusMap = { pending: '⏳ 待决', merged: '✅ 已融合', disambiguated: '🔀 已消歧', rejected: '❌ 已拒绝', resolved: '✅ 已处理' };
        const statusText = statusMap[item.status] || item.status;
        let actions = '';
        if (item.status === 'pending') {
          actions = `
            <button class="kb-action-btn kb-btn-merge" onclick="kbShowPendingDetail(${item.id},'merge')">融合</button>
            <button class="kb-action-btn kb-btn-disambig" onclick="kbShowPendingDetail(${item.id},'disambiguate')">消歧</button>
            <button class="kb-action-btn kb-btn-reject" onclick="kbShowPendingDetail(${item.id},'reject')">拒绝</button>`;
        } else {
          actions = `<span class="kb-status-${item.status}">${statusText}</span>`;
        }
        return `<tr>
          <td>${item.id}</td>
          <td>${item.entity_type}</td>
          <td title="${_esc(item.vid_a)}">${_esc(item.name_a) || '-'}</td>
          <td title="${_esc(item.vid_b)}">${_esc(item.name_b) || '-'}</td>
          <td><span class="${simClass}">${sim}</span></td>
          <td>${verdict}</td>
          <td>${actions}</td>
        </tr>`;
      }).join('');
    }

    // 分页
    const pages = Math.ceil(d.total / PENDING_PAGE_SIZE);
    const pag = document.getElementById('kb-pending-pagination');
    if (pages <= 1) {
      pag.innerHTML = '';
    } else {
      pag.innerHTML = `<span>第 ${_pendingPage + 1}/${pages} 页</span>
        <button ${_pendingPage === 0 ? 'disabled' : ''} onclick="kbLoadPending(${_pendingPage - 1})">上一页</button>
        <button ${_pendingPage >= pages - 1 ? 'disabled' : ''} onclick="kbLoadPending(${_pendingPage + 1})">下一页</button>`;
    }
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="7" style="color:#ef4444">加载失败: ${e.message}</td></tr>`;
  }
};

window.kbFusionAnalyze = async function() {
  const btn = document.getElementById('btn-fusion-analyze');
  btn.textContent = '⏳ 全量扫描中（含 LLM 判断）...'; btn.disabled = true;
  try {
    const r = await fetch(`${ADMIN}/fusion/analyze`, { method: 'POST' });
    const d = await r.json();
    btn.textContent = `✅ 自动融合 ${d.auto_merge || 0}，LLM判断 ${d.llm_judged || 0}，待决 ${d.pending || 0}`;
    setTimeout(() => { btn.textContent = '🔍 全量扫描'; btn.disabled = false; }, 5000);
    kbLoadPending();
    if (window.kbRefreshStats) kbRefreshStats();
  } catch (e) {
    btn.textContent = '❌ 失败';
    setTimeout(() => { btn.textContent = '🔍 全量扫描'; btn.disabled = false; }, 3000);
  }
};

// ─── 对比弹窗 ───

window.kbShowPendingDetail = async function(id, action) {
  const overlay = document.getElementById('pending-compare-overlay');
  const body = document.getElementById('pending-compare-body');
  overlay.style.display = 'flex';
  body.innerHTML = '<div style="text-align:center;padding:40px;color:#94a3b8">加载中...</div>';

  // 记住用户选的操作
  overlay.dataset.pendingId = id;
  overlay.dataset.action = action;

  try {
    const r = await fetch(`${ADMIN}/pending-entities/${id}/detail`);
    const d = await r.json();
    if (d.error) {
      body.innerHTML = `<div style="color:#ef4444;padding:20px">错误: ${d.error}</div>`;
      return;
    }
    _renderCompare(d, action);
  } catch (e) {
    body.innerHTML = `<div style="color:#ef4444;padding:20px">加载失败: ${e.message}</div>`;
  }
};

function _renderCompare(d, action) {
  window._pendingDetail = d; // store for merge action
  const body = document.getElementById('pending-compare-body');
  const labels = { merge: '融合（合并为同一实体）', disambiguate: '消歧（确认为不同实体）', reject: '拒绝（丢弃此记录）' };

  // 属性对比
  const allKeys = new Set([...Object.keys(d.props_a || {}), ...Object.keys(d.props_b || {})]);
  // 要跳过的 key（显示没意义或太长）
  const skipKeys = new Set(['created_at', 'updated_at', 'confidence']);
  const propRows = [...allKeys].filter(k => !skipKeys.has(k)).sort().map(k => {
    const va = _formatProp(d.props_a?.[k]);
    const vb = _formatProp(d.props_b?.[k]);
    const diff = va !== vb ? 'pd-diff' : '';
    return `<tr class="${diff}"><td class="pd-key">${_niceKey(k)}</td><td>${va}</td><td>${vb}</td></tr>`;
  }).join('');

  // 相关报道
  const artsA = (d.articles_a || []).map(a =>
    `<div class="pd-article"><a href="${_esc(a.url)}" target="_blank" rel="noopener">${_esc(a.title)}</a><span class="pd-date">${a.published_at || ''}</span></div>`
  ).join('') || '<span class="pd-empty">无</span>';
  const artsB = (d.articles_b || []).map(a =>
    `<div class="pd-article"><a href="${_esc(a.url)}" target="_blank" rel="noopener">${_esc(a.title)}</a><span class="pd-date">${a.published_at || ''}</span></div>`
  ).join('') || '<span class="pd-empty">无</span>';

  // 最长句子
  const sentA = d.longest_sentence_a ? `<div class="pd-sentence">"${_esc(d.longest_sentence_a)}"</div>` : '<span class="pd-empty">无</span>';
  const sentB = d.longest_sentence_b ? `<div class="pd-sentence">"${_esc(d.longest_sentence_b)}"</div>` : '<span class="pd-empty">无</span>';

  // LLM 分析
  let llmSection = '';
  if (d.llm_analysis && (d.llm_analysis.props_a || d.llm_analysis.props_b)) {
    llmSection = `<div class="pd-section"><div class="pd-section-title">🤖 LLM 分析数据</div><pre class="pd-pre">${_esc(JSON.stringify(d.llm_analysis, null, 2))}</pre></div>`;
  }
  if (d.notes) {
    llmSection += `<div class="pd-section"><div class="pd-section-title">📝 备注</div><div style="color:#94a3b8">${_esc(d.notes)}</div></div>`;
  }

  const sim = d.similarity ? (d.similarity * 100).toFixed(1) + '%' : '-';
  const verdictMap = { merge: '融合', disambiguate: '消歧', l2_uncertain: 'L2不确定', unclear: 'LLM不确定', auto_merge: '自动融合' };

  body.innerHTML = `
    <div class="pd-header">
      <div class="pd-title">实体对比 #${d.id}
        <span class="pd-sim">相似度 ${sim}</span>
        <span class="pd-verdict">${verdictMap[d.llm_verdict] || d.llm_verdict || ''}</span>
        <span class="pd-type">${d.entity_type}</span>
      </div>
      <div class="pd-action-label">操作：${labels[action] || action}</div>
    </div>
    <div class="pd-cols">
      <div class="pd-col">
        <div class="pd-col-title">实体 A：${_esc(d.name_a)}</div>
      </div>
      <div class="pd-col">
        <div class="pd-col-title">实体 B：${_esc(d.name_b)}</div>
      </div>
    </div>
    <div class="pd-section">
      <div class="pd-section-title">📊 属性对比</div>
      <table class="pd-table">
        <thead><tr><th style="width:120px">属性</th><th>实体 A</th><th>实体 B</th></tr></thead>
        <tbody>${propRows || '<tr><td colspan="3" class="pd-empty">无属性数据</td></tr>'}</tbody>
      </table>
    </div>
    <div class="pd-cols">
      <div class="pd-col">
        <div class="pd-section-title">📰 相关报道 (${(d.articles_a || []).length})</div>
        ${artsA}
      </div>
      <div class="pd-col">
        <div class="pd-section-title">📰 相关报道 (${(d.articles_b || []).length})</div>
        ${artsB}
      </div>
    </div>
    <div class="pd-cols">
      <div class="pd-col">
        <div class="pd-section-title">💬 最长提及句</div>
        ${sentA}
      </div>
      <div class="pd-col">
        <div class="pd-section-title">💬 最长提及句</div>
        ${sentB}
      </div>
    </div>
    ${llmSection}
    <div class="pd-footer">
      <button class="kb-action-btn kb-btn-merge" onclick="kbConfirmResolve('merge')">✅ 确认融合</button>
      <button class="kb-action-btn kb-btn-disambig" onclick="kbConfirmResolve('disambiguate')">🔀 确认消歧</button>
      <button class="kb-action-btn kb-btn-reject" onclick="kbConfirmResolve('reject')">❌ 拒绝</button>
      <button class="kb-action-btn" onclick="kbCloseCompare()" style="background:#374151">取消</button>
    </div>
  `;
}

// Map frontend action names to backend action names
const ACTION_MAP = { merge: 'merge', disambiguate: 'keep', reject: 'discard' };

window.kbConfirmResolve = async function(action) {
  const overlay = document.getElementById('pending-compare-overlay');
  const id = overlay.dataset.pendingId;
  if (!id) return;
  const labels = { merge: '融合', disambiguate: '消歧', reject: '拒绝' };

  // For merge: need to know which entity to keep
  let body = { action: ACTION_MAP[action] || action };

  if (action === 'merge') {
    const detail = window._pendingDetail;
    if (!detail || !detail.vid_a || !detail.vid_b) {
      alert('缺少实体VID信息，无法融合。请确保记录包含 vid_a 和 vid_b。');
      return;
    }
    const choice = confirm(
      `确认融合？\n\n点击「确定」= 保留实体A（${detail.name_a || detail.vid_a}）\n` +
      `点击「取消」= 保留实体B（${detail.name_b || detail.vid_b}）`
    );
    body.target_vid = choice ? detail.vid_a : detail.vid_b;
  } else {
    if (!confirm(`确认${labels[action]}？`)) return;
  }

  try {
    const resp = await fetch(`${ADMIN}/pending-entities/${id}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const result = await resp.json();
    if (result.status === 'error') {
      alert('操作失败: ' + result.message);
      return;
    }
    kbCloseCompare();
    kbLoadPending();
    if (window.kbRefreshStats) kbRefreshStats();
  } catch (e) {
    alert('操作失败: ' + e.message);
  }
};

window.kbCloseCompare = function() {
  document.getElementById('pending-compare-overlay').style.display = 'none';
};

// ─── 工具函数 ───

function _truncate(s, len) {
  if (!s) return '-';
  return s.length > len ? s.substring(0, len) + '...' : s;
}

function _esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function _formatProp(v) {
  if (v === null || v === undefined || v === '') return '<span class="pd-empty">-</span>';
  if (Array.isArray(v)) return v.join(', ') || '<span class="pd-empty">-</span>';
  if (typeof v === 'string' && v.startsWith('{') && v.endsWith('}')) {
    // Nebula set format
    const items = v.slice(1, -1).split(',').map(s => s.trim().replace(/"/g, '')).filter(Boolean);
    return items.join(', ') || '-';
  }
  return _esc(String(v));
}

function _niceKey(k) {
  const map = {
    name: '名称', equip_type: '装备类型', category: '分类', state: '状态',
    home_location: '驻地', parent_unit: '上级单位', occupation: '职业',
    org_name: '所属组织', event_type: '事件类型', start_date: '开始日期',
    end_date: '结束日期', location_name: '地点', loc_type: '地点类型',
    region: '区域', coordinates: '坐标', aliases: '别名',
    description: '描述', latest_reported_at: '最近报道',
  };
  return map[k] || k;
}
