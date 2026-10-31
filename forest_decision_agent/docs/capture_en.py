"""영문 화면(12·13) 다시 캡처: 영문 변화 해석, 영문 규칙 기반 보고서."""
import json, shutil, subprocess, tempfile, time, urllib.request
from capture_screens import BASE, CHROME, H, PORT, W, Tab
prof = tempfile.mkdtemp(prefix="cdp_")
p = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}", f"--window-size={W},{H}", "--hide-scrollbars", "about:blank"])
try:
    for _ in range(60):
        try:
            page = next(t for t in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read()) if t["type"] == "page"); break
        except Exception:
            time.sleep(0.5)
    tab = Tab(page["webSocketDebuggerUrl"]); tab.cmd("Page.enable"); tab.cmd("Runtime.enable")
    tab.cmd("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
    tab.go(f"{BASE}/?lang=en")
    fid = tab.js("L_['소반'].getLayers().find(l => l.feature.properties['단위ID']==='060-000/010-000').feature.properties.fid", False)
    tab.js(f"showAttr('소반', {fid}, true); 0")
    tab.wait("document.querySelector('#aEx')")
    tab.js(f"showExplain({fid}); 0")
    tab.wait("document.querySelector('#ex table')")
    tab.js("document.querySelector('#ex').scrollIntoView(); 0", False)
    tab.shot("12_attributes_en")
    tab.go(f"{BASE}/?lang=en")
    tab.js("const s=document.querySelector('#emdSel'); s.value='동면'; s.onchange({target:s});"
           "document.querySelector('#qText').value='Using forest-map survey history and yearly forest-type estimates, rank stands in Dong-myeon national forest for forest-map re-survey.';"
           "document.querySelector('#mode').value='rule'; document.querySelector('#runBtn').click(); 0")
    tab.wait("!document.querySelector('#rep').hidden && document.querySelector('.hllist')", 300)
    tab.shot("13_report_en", settle=6)
finally:
    p.terminate(); time.sleep(1); shutil.rmtree(prof, ignore_errors=True)
