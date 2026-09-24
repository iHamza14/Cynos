with open("index.html", "r") as f:
    content = f.read()

# Add Viterbi polyline
content = content.replace("let groundTruthGhost = L.polyline([], { color: '#94a3b8', weight: 3, opacity: 0.4, dashArray: '5, 5' }).addTo(map);",
"""let groundTruthGhost = L.polyline([], { color: '#94a3b8', weight: 3, opacity: 0.4, dashArray: '5, 5' }).addTo(map);
        let viterbiPolyline = L.polyline([], { color: '#ef4444', weight: 5, opacity: 0.9 }).addTo(map);""")

# Add to legend
content = content.replace('<div class="flex items-center"><div class="w-4 h-1 bg-blue-500 mr-2 rounded"></div> Model Inference (DVSE + GyroTCN)</div>',
'<div class="flex items-center"><div class="w-4 h-1 bg-blue-500 mr-2 rounded"></div> Raw Model Inference (DVSE + GyroTCN)</div>\n                <div class="flex items-center"><div class="w-4 h-1 bg-red-500 mr-2 rounded"></div> Viterbi Map-Matched Prediction</div>')

# Clear viterbi in game loop
content = content.replace("groundTruthGhost.setLatLngs([]);",
"groundTruthGhost.setLatLngs([]);\n                    viterbiPolyline.setLatLngs([]);")

# Add viterbi point in BLACKOUT phase
content = content.replace("inferencePolyline.addLatLng([currentPoint.lat, currentPoint.lon]);",
"""inferencePolyline.addLatLng([currentPoint.lat, currentPoint.lon]);
                
                // Add Viterbi point
                const vitPoint = currentData.viterbi[tick];
                if(vitPoint) {
                    viterbiPolyline.addLatLng([vitPoint.lat, vitPoint.lon]);
                    
                    // Actually pan to the Viterbi point since it's the final output
                    currentPoint = vitPoint;
                }""")

# Update the status text slightly to mention Viterbi
content = content.replace("engineStatus.textContent = 'DVSE + GyroTCN Running';", "engineStatus.textContent = 'DVSE + Viterbi Filter Running';")

with open("index.html", "w") as f:
    f.write(content)
