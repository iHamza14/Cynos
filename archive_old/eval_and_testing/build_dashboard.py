import json

with open("live_simulation_log.json", "r") as f:
    data = json.load(f)

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>IDR Live Simulation</title>
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
        <h2 class="text-lg font-bold">IDR Live Replay</h2>
        
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
            <p><strong>DR Speed:</strong> <span id="lblSpeed">0.00</span> m/s</p>
            <p><strong>Raw Drift:</strong> <span id="lblRaw" class="text-red-600">0.0</span> m</p>
            <p><strong>Snapped Drift:</strong> <span id="lblSnap" class="text-green-600">0.0</span> m</p>
            <p><strong>GNSS Status:</strong> <span id="lblGNSS" class="text-red-600 font-bold">BLACKOUT</span></p>
        </div>
        
        <div class="mt-2 text-xs text-gray-600">
            <div class="flex items-center gap-2"><div class="w-3 h-3 bg-red-500 rounded-full"></div> Raw DR Estimate</div>
            <div class="flex items-center gap-2"><div class="w-3 h-3 bg-green-500 rounded-full"></div> Viterbi Snapped DR</div>
            <div class="flex items-center gap-2"><div class="w-3 h-3 bg-blue-500 rounded-full"></div> Ground Truth</div>
            <div class="flex items-center gap-2"><div class="w-3 h-3 bg-black rounded-full"></div> Last GNSS Anchor</div>
        </div>
    </div>

    <script>
        const simData = {json.dumps(data)};
        const anchor = simData.anchor;
        const trajectory = simData.trajectory;
        
        const map = L.map('map').setView([anchor.lat, anchor.lon], 18);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }}).addTo(map);

        const gtTrail = L.polyline([], {{color: 'blue', weight: 4, opacity: 0.5}}).addTo(map);
        const snapTrail = L.polyline([], {{color: 'lime', weight: 4, opacity: 0.8}}).addTo(map);
        const drTrail = L.polyline([], {{color: 'red', weight: 4, dashArray: '5, 10', opacity: 0.8}}).addTo(map);
        
        const anchorMarker = L.circleMarker([anchor.lat, anchor.lon], {{color: 'black', radius: 6, fillOpacity: 1}}).addTo(map);
        anchorMarker.bindPopup("Last Trusted GNSS");
        
        const snapMarker = L.circleMarker([anchor.lat, anchor.lon], {{color: 'lime', radius: 5, fillOpacity: 1}}).addTo(map);
        const drMarker = L.circleMarker([anchor.lat, anchor.lon], {{color: 'red', radius: 5, fillOpacity: 1}}).addTo(map);
        const gtMarker = L.circleMarker([anchor.lat, anchor.lon], {{color: 'blue', radius: 5, fillOpacity: 1}}).addTo(map);

        let timer = null;
        let currentIndex = 0;
        let playbackSpeed = 1;
        const tickMs = 100;
        
        function updateUI() {{
            const frame = trajectory[currentIndex];
            if(!frame) return;
            
            drMarker.setLatLng([frame.dr_lat, frame.dr_lon]);
            snapMarker.setLatLng([frame.snap_lat, frame.snap_lon]);
            gtMarker.setLatLng([frame.gt_lat, frame.gt_lon]);
            
            const drPts = trajectory.slice(0, currentIndex+1).map(f => [f.dr_lat, f.dr_lon]);
            const snapPts = trajectory.slice(0, currentIndex+1).map(f => [f.snap_lat, f.snap_lon]);
            const gtPts = trajectory.slice(0, currentIndex+1).map(f => [f.gt_lat, f.gt_lon]);
            
            drTrail.setLatLngs(drPts);
            snapTrail.setLatLngs(snapPts);
            gtTrail.setLatLngs(gtPts);
            
            if (currentIndex % 20 === 0) {{
                map.panTo([frame.snap_lat, frame.snap_lon], {{animate: true, duration: 0.5}});
            }}
            
            document.getElementById('lblTime').innerText = frame.time.toFixed(1);
            document.getElementById('lblSpeed').innerText = frame.dr_speed.toFixed(2);
            document.getElementById('lblRaw').innerText = frame.raw_err.toFixed(1);
            document.getElementById('lblSnap').innerText = frame.snap_err.toFixed(1);
        }}
        
        function tick() {{
            if(currentIndex < trajectory.length - 1) {{
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
            map.setView([anchor.lat, anchor.lon], 18);
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

with open("live_dashboard.html", "w") as f:
    f.write(html_content)
print("Updated live_dashboard.html with Viterbi snapping!")
