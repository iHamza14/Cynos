"""
Utility module: build_dashboard_all.py.
"""

import json

with open("live_simulation_all.json", "r") as f:
    data = json.load(f)

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>IDR Live Simulation - All Episodes</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        #map {{ height: 100vh; width: 100%; }}
        .overlay {{
            position: absolute; top: 20px; right: 20px; z-index: 1000;
            background: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            width: 320px; font-family: sans-serif;
        }}
    </style>
</head>
<body class="m-0 p-0 overflow-hidden">
    <div id="map"></div>
    <div class="overlay flex flex-col gap-3">
        <h2 class="text-lg font-bold">Global Live Replay (15 Vehicles)</h2>
        
        <div class="flex gap-2">
            <button id="btnPlay" class="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700">Play</button>
            <button id="btnPause" class="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600">Pause</button>
            <button id="btnReset" class="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600">Reset</button>
        </div>
        
        <div>
            <label class="text-sm font-semibold">Speed: <span id="speedLabel">1x</span></label>
            <input type="range" id="speedSlider" min="1" max="10" value="1" class="w-full">
        </div>
        
        <div class="text-sm">
            <p><strong>Time:</strong> <span id="lblTime">0.0</span>s</p>
            <p><strong>Mean Raw Drift:</strong> <span id="lblRaw" class="text-red-600">0.0</span> m</p>
            <p><strong>Mean Snapped Drift:</strong> <span id="lblSnap" class="text-green-600">0.0</span> m</p>
            <p><strong>Status:</strong> <span id="lblGNSS" class="text-red-600 font-bold">MASS BLACKOUT</span></p>
        </div>
        
        <div class="mt-2 text-xs text-gray-600">
            <div class="flex items-center gap-2"><div class="w-3 h-3 bg-red-500 rounded-full"></div> Raw DR Estimate</div>
            <div class="flex items-center gap-2"><div class="w-3 h-3 bg-green-500 rounded-full"></div> Viterbi Snapped DR</div>
            <div class="flex items-center gap-2"><div class="w-3 h-3 bg-blue-500 rounded-full"></div> Ground Truth</div>
        </div>
    </div>

    <script>
        const allSims = {json.dumps(data)};
        
        const map = L.map('map').setView([allSims[0].anchor.lat, allSims[0].anchor.lon], 13);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }}).addTo(map);

        const vehicles = allSims.map(sim => {{
            return {{
                sim: sim,
                gtTrail: L.polyline([], {{color: 'blue', weight: 2, opacity: 0.3}}).addTo(map),
                snapTrail: L.polyline([], {{color: 'lime', weight: 3, opacity: 0.8}}).addTo(map),
                drTrail: L.polyline([], {{color: 'red', weight: 2, dashArray: '5, 10', opacity: 0.5}}).addTo(map),
                snapMarker: L.circleMarker([sim.anchor.lat, sim.anchor.lon], {{color: 'lime', radius: 4, fillOpacity: 1}}).addTo(map)
            }};
        }});

        let timer = null;
        let currentIndex = 0;
        let playbackSpeed = 1;
        const tickMs = 100;
        
        function updateUI() {{
            let totalRaw = 0;
            let totalSnap = 0;
            let validCount = 0;
            
            vehicles.forEach(v => {{
                const frame = v.sim.trajectory[currentIndex];
                if(!frame) return;
                
                v.snapMarker.setLatLng([frame.snap_lat, frame.snap_lon]);
                
                const snapPts = v.sim.trajectory.slice(0, currentIndex+1).map(f => [f.snap_lat, f.snap_lon]);
                const gtPts = v.sim.trajectory.slice(0, currentIndex+1).map(f => [f.gt_lat, f.gt_lon]);
                const drPts = v.sim.trajectory.slice(0, currentIndex+1).map(f => [f.dr_lat, f.dr_lon]);
                
                v.snapTrail.setLatLngs(snapPts);
                v.gtTrail.setLatLngs(gtPts);
                v.drTrail.setLatLngs(drPts);
                
                totalRaw += frame.raw_err;
                totalSnap += frame.snap_err;
                validCount++;
            }});
            
            if(validCount > 0) {{
                const timeStr = vehicles[0].sim.trajectory[currentIndex].time.toFixed(1);
                document.getElementById('lblTime').innerText = timeStr;
                document.getElementById('lblRaw').innerText = (totalRaw / validCount).toFixed(1);
                document.getElementById('lblSnap').innerText = (totalSnap / validCount).toFixed(1);
            }}
        }}
        
        function tick() {{
            const maxLen = vehicles[0].sim.trajectory.length;
            if(currentIndex < maxLen - 1) {{
                currentIndex++;
                updateUI();
            }} else {{
                pause();
            }}
        }}
        
        function play() {{
            if(timer) clearInterval(timer);
            timer = setInterval(tick, tickMs / playbackSpeed);
        }}
        
        function pause() {{
            if(timer) clearInterval(timer);
            timer = null;
        }}
        
        function reset() {{
            pause();
            currentIndex = 0;
            updateUI();
        }}
        
        document.getElementById('btnPlay').onclick = play;
        document.getElementById('btnPause').onclick = pause;
        document.getElementById('btnReset').onclick = reset;
        document.getElementById('speedSlider').oninput = (e) => {{
            playbackSpeed = e.target.value;
            document.getElementById('speedLabel').innerText = playbackSpeed + 'x';
            if(timer) play();
        }};
        
        updateUI();
    </script>
</body>
</html>
"""

with open("live_dashboard_all.html", "w") as f:
    f.write(html_content)
print("Updated live_dashboard_all.html for global fleet!")
