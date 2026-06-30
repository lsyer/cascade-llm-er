/**
 * 常量与配置
 */
export const API = new URL('./api', window.location.href).href.replace(/\/$/, '');

export const EQUIP_COLORS = {
  aircraft_carrier: '#3b82f6', destroyer: '#10b981', cruiser: '#f59e0b',
  submarine: '#ef4444', ssbn: '#ef4444', amphibious_assault: '#8b5cf6',
  littoral_combat: '#06b6d4', aircraft: '#22c55e', vehicle: '#a855f7',
  weapon: '#dc2626', uav: '#14b8a6', other: '#94a3b8',
};

export const TYPE_LABELS = {
  aircraft_carrier: '航母', destroyer: '驱逐舰', cruiser: '巡洋舰',
  submarine: '潜艇', ssbn: '战略潜艇', amphibious_assault: '两栖舰',
  littoral_combat: '濒海舰', aircraft: '飞机', vehicle: '车辆',
  weapon: '武器', uav: '无人机', other: '其他', unknown: '未知', ship: '舰船',
};

export const LOC_LABELS = {
  naval_station: '海军站', naval_air_station: '航空站', naval_base: '海军基地',
  air_base: '空军基地', sea_area: '海域', strait: '海峡', port: '港口',
  region: '地区', shipyard: '船厂', submarine_base: '潜艇基地', other: '其他',
  marine_air_station: '陆战队航空站',
};

export const ACT_LABELS = {
  operation: '军事行动', deployment: '部署', exercise: '演习',
  port_visit: '港口访问', transit: '过航', incident: '事件',
  surveillance: '监视', other: '其他',
};
