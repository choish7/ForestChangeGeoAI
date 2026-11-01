"""의사결정 웹서비스 화면 캡처 (Chrome DevTools Protocol, 헤드리스 크롬).
  python forest_decision_agent/docs/capture_screens.py   (서버가 http://127.0.0.1:8800 에서 실행 중이어야 함)
출력: forest_decision_agent/docs/screens/*.png
"""
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE = "http://127.0.0.1:8800"
OUT = Path(__file__).resolve().parent / "screens"
PORT = 9333
W, H = 1680, 960
RUN = "20261030_221546"          # 통합 질의 사례 보고서


class Tab:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=120, suppress_origin=True)
        self.i = 0

    def cmd(self, method, **params):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.i:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    def js(self, expr, await_promise=True):
        r = self.cmd("Runtime.evaluate", expression=expr, awaitPromise=await_promise, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r.get("result", {}).get("value")

    def wait(self, cond, timeout=90, step=0.5):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.js(f"!!({cond})", False):
                    return True
            except RuntimeError:
                pass
            time.sleep(step)
        raise TimeoutError(cond)

    def go(self, url):
        self.cmd("Page.navigate", url=url)
        time.sleep(1)
        self.wait("window.__ready === true", 120)
        time.sleep(3)                                   # 배경지도 타일

    def shot(self, name, settle=2.5):
        time.sleep(settle)
        d = self.cmd("Page.captureScreenshot", format="png", captureBeyondViewport=False)["data"]
        (OUT / f"{name}.png").write_bytes(base64.b64decode(d))
        print("캡처", name, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="cdp_")
    p = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}",
                          f"--window-size={W},{H}", "--hide-scrollbars", "--force-device-scale-factor=1", "--lang=ko-KR", "about:blank"])
    try:
        for _ in range(60):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read())
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        tab = Tab(page["webSocketDebuggerUrl"])
        tab.cmd("Page.enable")
        tab.cmd("Runtime.enable")
        tab.cmd("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)

        # 1 메인
        tab.go(f"{BASE}/?lang=ko")
        tab.shot("01_main_ko")
        # 2 장소 검색
        tab.js("tab('s'); document.querySelector('#sq').value='추곡리'; doSearch();")
        tab.wait("document.querySelectorAll('#sres .res').length > 0")
        tab.shot("02_search")
        # 3 속성 + 시계열 + 변화 해석 (060-000/010-000)
        fid = tab.js("L_['소반'].getLayers().find(l => l.feature.properties['단위ID']==='060-000/010-000').feature.properties.fid", False)
        tab.js(f"showAttr('소반', {fid}, true)")
        tab.wait("document.querySelector('#aTs')")
        tab.shot("03_attributes")
        tab.js(f"showTs('소반', {fid})")
        tab.wait("document.querySelector('#ts svg')")
        tab.js(f"showExplain({fid})")
        tab.wait("document.querySelector('#ex table')")
        tab.js("document.querySelector('#ex').scrollIntoView()", False)
        tab.shot("04_explain_timeseries")
        tab.js("document.querySelector('#ts').scrollIntoView()", False)
        tab.shot("05_timeseries")
        # 6 유사 소반
        tab.js(f"showSimilar({fid})")
        tab.wait("document.querySelector('#ex table')")
        tab.js("document.querySelector('#ex').scrollIntoView()", False)
        tab.shot("06_similar")
        # 7 지역 요약 (동면)
        tab.js("clearResult(); const s=document.querySelector('#emdSel'); s.value='동면'; s.onchange({target:s}); tab('r'); document.querySelector('#sumBtn').click();")
        tab.wait("document.querySelector('#sum table')")
        tab.shot("07_area_summary")
        # 8 저장 보고서 + 결과 강조
        tab.js(f"tab('h'); loadRuns(); openRun('{RUN}');")
        tab.wait("document.querySelector('.hllist')")
        tab.shot("08_report_highlight", settle=5)
        # 9 질의 실행(규칙 기반, 동면, 재조사)
        tab.go(f"{BASE}/?lang=ko")
        tab.js("const s=document.querySelector('#emdSel'); s.value='동면'; s.onchange({target:s});"
               "document.querySelector('#qText').value='임상도 조사 이력과 연도별 임상 추정을 바탕으로 동면 국유림의 임상도 재조사 우선순위를 정리해줘';"
               "document.querySelector('#mode').value='rule'; document.querySelector('#runBtn').click();")
        tab.wait("!document.querySelector('#rep').hidden && document.querySelector('.hllist')", 300)
        tab.shot("09_query_rule", settle=6)
        # 10 레이어 펼침 + 위성영상 + 모델 패치
        tab.js("map.removeLayer(osm); sat.addTo(map); L_['모델'].addTo(map); clearResult();"
               "document.querySelector('.leaflet-control-layers').classList.add('leaflet-control-layers-expanded');"
               "map.setView([38.0, 127.83], 13); 0", False)
        tab.shot("10_layers_satellite", settle=6)
        # 11~12 영문
        tab.go(f"{BASE}/?lang=en")
        tab.shot("11_main_en")
        tab.js(f"showAttr('소반', {fid}, true)")
        tab.wait("document.querySelector('#aEx')")
        tab.js(f"showExplain({fid})")
        tab.wait("document.querySelector('#ex table')")
        tab.js("document.querySelector('#ex').scrollIntoView()", False)
        tab.shot("12_attributes_en")
    finally:
        p.terminate()
        time.sleep(1)
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    main()
