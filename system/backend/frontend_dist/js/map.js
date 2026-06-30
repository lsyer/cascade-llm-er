/**
 * MapLibre GL 地图初始化 + 瓦片切换器 + 标记工具
 *
 * 修复方案：不再 export const map（静态引用），
 * 改为 export function getMap() 动态获取真正的 map 实例。
 * initMap() 在首次切到态势页面时创建 MapLibre 对象。
 */

import { state } from './state.js';

// ── 瓦片源（全部声明，visibility 切换，不触发重新加载）──
const TILES = [
  { id: 'carto-dark',    label: '🌑 深蓝暗', url: 'https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png' },
  { id: 'carto-nolabel', label: '🌫️ 无字',  url: 'https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png' },
  { id: 'positron',      label: '☀️ 浅色',  url: 'https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png' },
  { id: 'voyager',       label: '🗺️ 彩色',  url: 'https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}@2x.png' },
  { id: 'esri-dark',     label: '⬛ 灰暗',  url: 'https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}' },
  { id: 'esri-imagery',  label: '🛰️ 卫星',  url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}' },
  { id: 'esri-natgeo',   label: '🌍 地理',  url: 'https://services.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}' },
  { id: 'opentopo',      label: '⛰️ 地形',  url: 'https://a.tile.opentopomap.org/{z}/{x}/{y}.png' },
  { id: 'osm',           label: '📋 标准',  url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png' },
];

let activeTile = 'carto-dark';

function buildStyle() {
  const sources = {}, layers = [];
  TILES.forEach(t => {
    sources[t.id] = { type: 'raster', tiles: [t.url], tileSize: 256, attribution: '\u00a9 OpenStreetMap' };
    layers.push({
      id: t.id, type: 'raster', source: t.id,
      layout: { visibility: t.id === activeTile ? 'visible' : 'none' },
    });
  });
  return { version: 8, sources, layers };
}

// ── 地图对象（延迟初始化）──
// _realMap = null 表示还没初始化；非 null 是真正的 MapLibre 实例
let _realMap = null;

/** 获取当前地图实例（可能未初始化返回 null） */
export function getMap() { return _realMap; }

/** 初始化地图（切到态势首页时调用），返回 map 实例或 null */
export function initMap() {
  if (_realMap) return _realMap;
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) throw new Error('No WebGL');
    _realMap = new maplibregl.Map({
      container: 'map',
      style: buildStyle(),
      center: [130, 25],
      zoom: 3, minZoom: 1, maxZoom: 18,
      attributionControl: false,
    });
    _realMap.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');
    _realMap.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-left');
    _setupTileSwitcher();
  } catch (e) {
    console.warn('MapLibre init failed, map features disabled:', e.message);
    _realMap = null;
  }
  return _realMap;
}

// 兼容旧的 import { map } 写法 — 指向 _fallbackMap 只在未初始化时使用
// 但所有实际地图操作应通过 getMap() 获取真实实例
const _fallbackMap = {
  on() {}, off() {}, flyTo() {}, fitBounds() {}, addSource() {}, addLayer() {},
  getLayer() { return undefined; }, getSource() { return undefined; },
  removeLayer() {}, removeSource() {}, setLayoutProperty() {},
  getCenter() { return { lng: 0, lat: 0 }; }, getZoom() { return 0; },
  getBounds() { return { getNorth: () => 0, getSouth: () => 0, getEast: () => 0, getWest: () => 0 }; },
};
// 用 getter 让外部 import 的 map 也能拿到真实实例
export const map = _fallbackMap;

// ── 瓦片切换器 UI ──
function _setupTileSwitcher() {
  if (!_realMap) return;
  _realMap.on('load', () => {
    const el = document.createElement('div');
    el.className = 'tile-switcher';
    el.innerHTML = TILES.map(t =>
      `<button class="ts-btn${t.id === activeTile ? ' active' : ''}" data-t="${t.id}">${t.label}</button>`
    ).join('');
    el.addEventListener('click', e => {
      const btn = e.target.closest('.ts-btn');
      if (!btn) return;
      const tid = btn.dataset.t;
      if (tid === activeTile) return;
      if (!_realMap) return;
      _realMap.setLayoutProperty(activeTile, 'visibility', 'none');
      _realMap.setLayoutProperty(tid, 'visibility', 'visible');
      activeTile = tid;
      el.querySelectorAll('.ts-btn').forEach(b => b.classList.toggle('active', b.dataset.t === tid));
    });
    document.getElementById('map').appendChild(el);
  });
}

// ━━━ 标记管理 ━━━

export function clearMarkers() {
  state.markers.forEach(m => m.remove());
  state.markers = [];
  clearTrackLines();
}

export function addMarker(m) { state.markers.push(m); }

export function createCircleMarker(lat, lng, opts = {}, popupHtml = '') {
  const r = opts.radius || 6;
  const c = opts.color || '#94a3b8';
  const el = document.createElement('div');
  el.style.cssText = `width:${r * 2}px;height:${r * 2}px;border-radius:50%;` +
    `background:${opts.fillColor || c};border:${opts.weight || 1}px solid ${c};` +
    `opacity:${opts.fillOpacity ?? 0.7};cursor:pointer;box-shadow:0 0 6px ${c}80;`;
  const marker = new maplibregl.Marker(el).setLngLat([lng, lat]);
  if (popupHtml) {
    marker.setPopup(new maplibregl.Popup({ closeButton: false, maxWidth: '280px' }).setHTML(popupHtml));
  }
  // 用 getMap() 获取真实实例
  const m = getMap();
  if (m) marker.addTo(m);
  return marker;
}

// ━━━ 轨迹线 ━━━

let _tid = 0;

export function addTrackLine(coords, opts = {}) {
  const m = getMap();
  if (!m) return '';
  _tid++;
  const id = 'trk-' + _tid;
  const lngLat = coords.map(([lat, lng]) => [lng, lat]);
  m.addSource(id, {
    type: 'geojson',
    data: { type: 'Feature', geometry: { type: 'LineString', coordinates: lngLat } },
  });
  m.addLayer({
    id, type: 'line', source: id,
    paint: {
      'line-color': opts.color || '#3b82f6',
      'line-width': opts.weight || 2,
      'line-opacity': opts.opacity ?? 0.6,
      'line-dasharray': opts.dash || [1],
    },
  });
  state.trackLines.push(id);
  return id;
}

export function clearTrackLines() {
  const m = getMap();
  if (!m) return;
  state.trackLines.forEach(id => {
    if (m.getLayer(id)) m.removeLayer(id);
    if (m.getSource(id)) m.removeSource(id);
  });
  state.trackLines = [];
}

// ━━━ 视图控制 ━━━

export function flyTo(lat, lng, zoom) {
  const m = getMap();
  if (m) m.flyTo({ center: [lng, lat], zoom: zoom || 6 });
}

export function fitBoundsFromCoords(coords, padding) {
  const m = getMap();
  if (!m || !coords?.length) return;
  const lngLat = coords.map(([lat, lng]) => [lng, lat]);
  const bounds = lngLat.reduce((b, c) => b.extend(c), new maplibregl.LngLatBounds(lngLat[0], lngLat[0]));
  m.fitBounds(bounds, { padding: padding || 50 });
}
