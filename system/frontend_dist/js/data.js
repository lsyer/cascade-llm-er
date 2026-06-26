/**
 * 数据加载
 */
import { API } from './config.js';
import { state } from './state.js';
import { renderAll } from './render.js';

export async function loadData() {
  try {
    const resp = await fetch(`${API}/map/overview`);
    const data = await resp.json();

    state.allData.equipment = data.equipment || [];
    state.allData.locations = data.locations || [];
    state.allData.activities = data.activities || [];

    document.getElementById('stats').textContent =
      `${data.stats?.total_equipment || 0} 器装(舰${data.stats?.total_ships || 0} 机${data.stats?.total_aircraft || 0} 武器${data.stats?.total_weapons || 0}) · ${data.stats?.total_locations || 0} 位置 · ` +
      `${data.stats?.equipment_with_pos || 0} 已定位 · ${data.stats?.articles_7d || 0} 近7日报道`;

    renderAll();
  } catch (e) {
    console.error('Failed to load data:', e);
    document.getElementById('stats').textContent = '加载失败';
  }
}

export async function loadPersons() {
  if (state.allData.persons.length) return;
  try {
    const resp = await fetch(`${API}/persons?limit=200`);
    state.allData.persons = await resp.json();
  } catch (e) {
    console.error('Failed to load persons:', e);
  }
}

export async function loadArticles(offset = 0, status = null) {
  try {
    let url = `${API}/articles?limit=50&offset=${offset}`;
    if (status) url += `&status=${status}`;
    const resp = await fetch(url);
    const data = await resp.json();
    return data; // { total, items }
  } catch (e) {
    console.error('Failed to load articles:', e);
    return { total: 0, items: [] };
  }
}
window.loadData = loadData;
