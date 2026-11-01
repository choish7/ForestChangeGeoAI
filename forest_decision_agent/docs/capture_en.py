"""영문 화면(12·13) 다시 캡처: 영문 변화 해석, 영문 규칙 기반 보고서.
  --hi : 2배 해상도(deviceScaleFactor 2) → screens_hi/ (논문 그림용)
"""
import json, shutil, subprocess, sys, tempfile, time, urllib.request
import capture_screens as CS
SAT = "--sat" in sys.argv          # 배경지도: Esri 위성영상
from capture_screens import BASE, CHROME, H, PORT, W, Tab
DSF = 2 if "--hi" in sys.argv else 1
if DSF == 2:
    CS.OUT = CS.OUT.parent / "screens_hi"
    CS.OUT.mkdir(exist_ok=True)
prof = tempfile.mkdtemp(prefix="cdp_")
p = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}", f"--window-size={W},{H}", "--hide-scrollbars", "about:blank"])
try:
    for _ in range(60):
        try:
            page = next(t for t in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read()) if t["type"] == "page"); break
        except Exception:
            time.sleep(0.5)
    tab = Tab(page["webSocketDebuggerUrl"]); tab.cmd("Page.enable"); tab.cmd("Runtime.enable")
    tab.cmd("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=DSF, mobile=False)
    tab.go(f"{BASE}/?lang=en")
    if SAT:
        tab.js("map.removeLayer(osm); sat.addTo(map); 0", False)
    fid = tab.js("L_['소반'].getLayers().find(l => l.feature.properties['단위ID']==='060-000/010-000').feature.properties.fid", False)
    tab.js(f"showAttr('소반', {fid}, true); 0")
    tab.wait("document.querySelector('#aEx')")
    tab.js(f"showExplain({fid}); 0")
    tab.wait("document.querySelector('#ex table')")
    tab.js("document.querySelector('#ex').scrollIntoView(); 0", False)
    if SAT:   # 위성영상이 보이도록 소반 채움을 반투명으로, 판정 색은 테두리로
        tab.js("L_['소반'].setStyle(f => ({...styleUnit(f), color: JC[f.properties['판정']] || '#ccc', weight: 2, fillOpacity: 0.18})); 0", False)
    tab.shot("12_attributes_en", settle=6 if SAT else 2.5)
    tab.go(f"{BASE}/?lang=en")
    if SAT:
        tab.js("map.removeLayer(osm); sat.addTo(map); 0", False)
    tab.js("const s=document.querySelector('#emdSel'); s.value='동면'; s.onchange({target:s});"
           "document.querySelector('#qText').value='Using forest-map survey history and yearly forest-type estimates, rank stands in Dong-myeon national forest for forest-map re-survey.';"
           "document.querySelector('#mode').value='rule'; document.querySelector('#runBtn').click(); 0")
    tab.wait("!document.querySelector('#rep').hidden && document.querySelector('.hllist')", 300)
    tab.js("document.querySelector('#log').hidden = true; 0", False)   # 진행 로그(한글 도구명) 숨김
    tab.shot("13_report_en", settle=8 if SAT else 6)
finally:
    p.terminate(); time.sleep(1); shutil.rmtree(prof, ignore_errors=True)
