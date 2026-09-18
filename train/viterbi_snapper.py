#!/usr/bin/env python3
"""
Viterbi Snapper (Map Matching) for GNSS / Dead Reckoning Trajectories.

Implements the Hidden Markov Model (HMM) map matching algorithm based on:
Newson, P. and Krumm, J. (2009). Hidden Markov Map Matching Through Noise and Sparseness.
Reference: https://medium.com/gett-engineering/map-matching-the-viterbi-snapper-e32d11f0d130

It uses log-probabilities to avoid arithmetic underflow over long trajectories.
"""

import math
import numpy as np

EARTH_RADIUS = 6371000.0

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * EARTH_RADIUS * math.atan2(math.sqrt(a), math.sqrt(1-a))

class ViterbiSnapper:
    def __init__(self, sigma_z=10.0, beta=20.0):
        """
        Args:
            sigma_z: Standard deviation of GPS/DR measurement noise (meters).
                     For GNSS this is typically 4.0 - 10.0.
                     For Dead Reckoning (which accumulates drift), you may need a larger value.
            beta: Transition probability parameter (meters). Represents the expected mismatch
                  between great-circle distance and actual route distance.
        """
        self.sigma_z = sigma_z
        self.beta = beta

    def log_emission_prob(self, distance):
        """
        Log likelihood of the observation given the candidate point.
        Modeled as a Zero-mean Gaussian distribution over the projection distance.
        """
        return -np.log(np.sqrt(2 * math.pi) * self.sigma_z) - 0.5 * (distance / self.sigma_z)**2

    def log_transition_prob(self, great_circle_dist, route_dist):
        """
        Log likelihood of transitioning between two candidate points.
        Modeled as an Exponential distribution over the absolute difference between 
        the direct observation distance and the road network route distance.
        """
        diff = abs(great_circle_dist - route_dist)
        return -np.log(self.beta) - (diff / self.beta)

    def snap(self, observations, get_candidates_fn, get_route_dist_fn):
        """
        Executes the Viterbi algorithm to find the most likely sequence of map-matched points.
        
        Args:
            observations: List of (lat, lon) tuples.
            get_candidates_fn: Function that takes an observation (lat, lon) and returns a list 
                               of Candidate dictionaries. Each candidate must have:
                               {'id': str, 'lat': float, 'lon': float, 'dist_to_obs': float}
            get_route_dist_fn: Function that takes (candidate1, candidate2) and returns the 
                               travel distance along the road network in meters.
        
        Returns:
            The sequence of matched candidates (the most probable path).
        """
        if not observations:
            return []

        # V[t][candidate_id] = (log_prob, previous_candidate_id)
        V = [{} for _ in range(len(observations))]

        # --- Initialization (t = 0) ---
        candidates_0 = get_candidates_fn(observations[0])
        if not candidates_0:
            raise ValueError("No candidates found for the initial observation.")
            
        for cand in candidates_0:
            # Initial state probability is just the emission probability
            log_p = self.log_emission_prob(cand['dist_to_obs'])
            V[0][cand['id']] = (log_p, None, cand)

        # --- Recursion (t = 1 to N-1) ---
        for t in range(1, len(observations)):
            obs_prev = observations[t-1]
            obs_curr = observations[t]
            
            candidates_curr = get_candidates_fn(obs_curr)
            if not candidates_curr:
                print(f"Warning: No candidates found for observation {t}: {obs_curr}. Path broken.")
                break
                
            gc_dist = haversine(obs_prev[0], obs_prev[1], obs_curr[0], obs_curr[1])
            
            for c_curr in candidates_curr:
                max_log_p = -float('inf')
                best_prev_id = None
                
                emission = self.log_emission_prob(c_curr['dist_to_obs'])
                
                # Compare against all candidates from the previous time step
                for prev_id, (prev_log_p, _, c_prev) in V[t-1].items():
                    route_dist = get_route_dist_fn(c_prev, c_curr)
                    transition = self.log_transition_prob(gc_dist, route_dist)
                    
                    joint_log_p = prev_log_p + transition + emission
                    
                    if joint_log_p > max_log_p:
                        max_log_p = joint_log_p
                        best_prev_id = prev_id
                        
                V[t][c_curr['id']] = (max_log_p, best_prev_id, c_curr)

        # --- Termination & Backtracking ---
        # Find the highest probability state at the final time step
        last_t = len(V) - 1
        while last_t >= 0 and not V[last_t]:
            last_t -= 1 # Fallback if path broke early
            
        if last_t < 0:
            return []

        best_final_id = max(V[last_t].keys(), key=lambda cid: V[last_t][cid][0])
        
        # Trace back the optimal path
        matched_path = []
        curr_id = best_final_id
        for t in range(last_t, -1, -1):
            _, prev_id, cand = V[t][curr_id]
            matched_path.append(cand)
            curr_id = prev_id
            
        matched_path.reverse()
        return matched_path


# ==============================================================================
# Helper classes for testing / demonstration without a full database like PostGIS
# ==============================================================================

class SimpleRoadGraph:
    """A minimal mock road network for testing the Viterbi Snapper."""
    def __init__(self, polylines):
        """polylines: list of list of (lat, lon) tuples representing road segments"""
        self.segments = []
        for line_id, line in enumerate(polylines):
            for i in range(len(line)-1):
                self.segments.append({
                    'id': f"road_{line_id}_seg_{i}",
                    'a': line[i],
                    'b': line[i+1]
                })

    def _project_point(self, p, a, b):
        """Projects point p onto segment a-b in local metric space. Returns (lat, lon, dist_m)."""
        # Convert to local Cartesian approximation (meters)
        lat_scale = 111320.0
        lon_scale = 111320.0 * math.cos(math.radians((a[0]+b[0]+p[0])/3))
        
        px, py = p[1]*lon_scale, p[0]*lat_scale
        ax, ay = a[1]*lon_scale, a[0]*lat_scale
        bx, by = b[1]*lon_scale, b[0]*lat_scale
        
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        
        # Projection of AP onto AB
        ab_squared = abx**2 + aby**2
        if ab_squared == 0:
            return a[0], a[1], haversine(p[0], p[1], a[0], a[1])
            
        t = max(0.0, min(1.0, (apx*abx + apy*aby) / ab_squared))
        
        proj_x = ax + t * abx
        proj_y = ay + t * aby
        
        proj_lon = proj_x / lon_scale
        proj_lat = proj_y / lat_scale
        
        dist = haversine(p[0], p[1], proj_lat, proj_lon)
        return proj_lat, proj_lon, dist

    def get_candidates(self, obs, radius=50.0):
        candidates = []
        for seg in self.segments:
            plat, plon, dist = self._project_point(obs, seg['a'], seg['b'])
            if dist <= radius:
                candidates.append({
                    'id': f"{seg['id']}_{plat:.6f}_{plon:.6f}",
                    'seg_id': seg['id'],
                    'lat': plat,
                    'lon': plon,
                    'dist_to_obs': dist
                })
        # If no segments are within radius, expand to nearest
        if not candidates:
            best_plat, best_plon, best_dist, best_seg = None, None, float('inf'), None
            for seg in self.segments:
                plat, plon, dist = self._project_point(obs, seg['a'], seg['b'])
                if dist < best_dist:
                    best_plat, best_plon, best_dist, best_seg = plat, plon, dist, seg
            candidates.append({
                'id': f"{best_seg['id']}_{best_plat:.6f}_{best_plon:.6f}",
                'seg_id': best_seg['id'],
                'lat': best_plat,
                'lon': best_plon,
                'dist_to_obs': best_dist
            })
        return candidates

    def route_distance(self, cand1, cand2):
        """
        Mock route distance. In a real engine (like OSRM/Valhalla), this queries the routing graph.
        Here we approximate with Euclidean/Haversine distance between the two projected points.
        """
        return haversine(cand1['lat'], cand1['lon'], cand2['lat'], cand2['lon'])


if __name__ == "__main__":
    print("Testing Viterbi Snapper...")
    
    # Define a straight road
    mock_road = [
        [(0.000, 0.000), (0.001, 0.000), (0.002, 0.000), (0.003, 0.000)]
    ]
    graph = SimpleRoadGraph(mock_road)
    
    # Simulate a noisy trajectory drifting off the road (e.g. from Dead Reckoning)
    noisy_observations = [
        (0.0000, 0.0001), # slightly off
        (0.0008, 0.0003), # drifting right
        (0.0015, 0.0005), # further right
        (0.0028, 0.0002), # coming back
    ]
    
    snapper = ViterbiSnapper(sigma_z=100.0, beta=20.0) # High sigma since it's DR, not clean GPS
    
    matched_path = snapper.snap(
        noisy_observations,
        get_candidates_fn=lambda obs: graph.get_candidates(obs, radius=200.0),
        get_route_dist_fn=lambda c1, c2: graph.route_distance(c1, c2)
    )
    
    print("\nObservation -> Snapped Result:")
    for obs, matched in zip(noisy_observations, matched_path):
        print(f"Obs: {obs[0]:.5f}, {obs[1]:.5f} -> Snapped: {matched['lat']:.5f}, {matched['lon']:.5f} (Dist: {matched['dist_to_obs']:>5.1f}m)")
