// app/static/app.js

// Mood gradients configuration
const MOODS = {
  energetic: ['#ff4d2e', '#ffb800'],
  chill: ['#1fa2a6', '#2d5bff'],
  moody: ['#3a3f58', '#6b5b95'],
  romantic: ['#d6336c', '#ff8fa3']
};

// State variables
let activeAudio = null;
let activeCard = null;
let currentPlaylist = [];
let recentSearches = JSON.parse(localStorage.getItem('moodify_searches') || '[]');

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const moodField = document.getElementById('moodField');
const cardsGrid = document.getElementById('cardsGrid');
const resultsLabel = document.getElementById('resultsLabel');
const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const weightsPanel = document.getElementById('weightsPanel');
const hybridToggle = document.getElementById('hybridToggle');
const recentSearchesContainer = document.getElementById('recentSearches');
const dbStats = document.getElementById('dbStats');

// Sliders and values
const alphaSlider = document.getElementById('alphaSlider');
const alphaVal = document.getElementById('alphaVal');
const betaSlider = document.getElementById('betaSlider');
const betaVal = document.getElementById('betaVal');
const gammaSlider = document.getElementById('gammaSlider');
const gammaVal = document.getElementById('gammaVal');
const limitSlider = document.getElementById('limitSlider');
const limitVal = document.getElementById('limitVal');

// Bottom Player
const bottomPlayer = document.getElementById('bottomPlayer');
const miniPlayBtn = document.getElementById('miniPlayBtn');
const playerTitle = document.getElementById('playerTitle');
const playerArtist = document.getElementById('playerArtist');
const currentTimeText = document.getElementById('currentTime');
const totalTimeText = document.getElementById('totalTime');
const playerScrub = document.getElementById('playerScrub');
const playerScrubFill = document.getElementById('playerScrubFill');

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
  fetchStats();
  renderRecentSearches();
  setupEventListeners();
  // Set default theme to chill
  setMoodTheme('chill');
});

// Setup Events
function setupEventListeners() {
  // Search button click
  searchBtn.addEventListener('click', () => {
    runSearch(searchInput.value);
  });

  // Search input enter key
  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      runSearch(searchInput.value);
    }
  });

  // Mood chips click
  document.querySelectorAll('#moodRow .chip').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#moodRow .chip').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mood = btn.dataset.mood;
      setMoodTheme(mood);
      fetchMoodRecommendations(mood);
    });
  });

  // Hybrid toggle visibility handler
  hybridToggle.addEventListener('change', () => {
    weightsPanel.style.display = hybridToggle.checked ? 'flex' : 'none';
  });

  // Slider value bubble updates
  alphaSlider.addEventListener('input', () => alphaVal.innerText = parseFloat(alphaSlider.value).toFixed(2));
  betaSlider.addEventListener('input', () => betaVal.innerText = parseFloat(betaSlider.value).toFixed(2));
  gammaSlider.addEventListener('input', () => gammaVal.innerText = parseFloat(gammaSlider.value).toFixed(2));
  limitSlider.addEventListener('input', () => limitVal.innerText = limitSlider.value);

  // Bottom player interaction
  miniPlayBtn.addEventListener('click', togglePlayState);
  
  // Scrubbing on progress bar
  playerScrub.addEventListener('click', (e) => {
    if (activeAudio) {
      const rect = playerScrub.getBoundingClientRect();
      const pct = (e.clientX - rect.left) / rect.width;
      activeAudio.currentTime = pct * activeAudio.duration;
    }
  });
}

// Fetch stats on load
async function fetchStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    dbStats.innerText = `${data.total_tracks.toLocaleString()} tracks indexed · Engines: ${data.engines}`;
    if (!data.hybrid_available) {
      hybridToggle.disabled = true;
      document.querySelector('.toggle-container').style.opacity = '0.5';
      document.querySelector('.toggle-container').title = 'Two-tower model weights not found';
    }
  } catch (err) {
    console.error('Failed to load DB stats', err);
  }
}

// Map text query to color theme
function detectMoodFromQuery(query) {
  const q = query.toLowerCase();
  if (anyInString(q, ["gym", "workout", "run", "hype", "pump", "beast", "intense", "power", "cardio", "party", "club", "dance", "disco", "rave", "edm", "banger", "happy", "hustle"])) {
    return 'energetic';
  }
  if (anyInString(q, ["sad", "heartbreak", "moody", "dark", "melancholy", "depressed", "lonely", "ambient", "rainy", "cry", "pain"])) {
    return 'moody';
  }
  if (anyInString(q, ["romantic", "love", "dinner", "date", "sweet", "passion", "hug"])) {
    return 'romantic';
  }
  return 'chill';
}

function anyInString(str, keywords) {
  return keywords.some(k => str.includes(k));
}

// Set CSS variables for background gradient
function setMoodTheme(mood) {
  const colors = MOODS[mood] || MOODS.chill;
  document.documentElement.style.setProperty('--mood-a', colors[0]);
  document.documentElement.style.setProperty('--mood-b', colors[1]);
}

// Save & Render recent searches
function saveSearch(query) {
  if (!query || query.trim() === '') return;
  const q = query.trim();
  recentSearches = recentSearches.filter(s => s.toLowerCase() !== q.toLowerCase());
  recentSearches.unshift(q);
  recentSearches = recentSearches.slice(0, 5); // limit to 5
  localStorage.setItem('moodify_searches', JSON.stringify(recentSearches));
  renderRecentSearches();
}

function renderRecentSearches() {
  recentSearchesContainer.innerHTML = '';
  if (recentSearches.length === 0) {
    recentSearchesContainer.innerHTML = '<span style="font-size:10px; opacity:0.5;">No recent searches</span>';
    return;
  }
  recentSearchesContainer.innerHTML = recentSearches.map(q => 
    `<button class="recent-chip" onclick="searchInput.value='${q.replace(/'/g, "\\'")}'; runSearch('${q.replace(/'/g, "\\\'")}')">${q}</button>`
  ).join('');
}

// Run searches
async function runSearch(query) {
  if (!query || query.trim() === '') return;
  saveSearch(query);
  
  // Update theme based on query
  const detectedMood = detectMoodFromQuery(query);
  setMoodTheme(detectedMood);
  // Clear active chip
  document.querySelectorAll('#moodRow .chip').forEach(b => b.classList.remove('active'));

  showLoading(true);
  
  const topK = limitSlider.value;
  const useHybrid = hybridToggle.checked;
  const alpha = alphaSlider.value;
  const beta = betaSlider.value;
  const gamma = gammaSlider.value;

  try {
    const url = `/api/search?q=${encodeURIComponent(query)}&top_k=${topK}&use_hybrid=${useHybrid}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Search failed');
    const data = await res.json();
    
    renderResults(data.tracks, `Recommendations based on ${data.label}`, data.engine_name, data.elapsed);
  } catch (err) {
    showError(err.message);
  }
}

// Fetch mood popular recommendations
async function fetchMoodRecommendations(mood) {
  showLoading(true);
  try {
    const res = await fetch(`/api/recommendations/mood/${mood}`);
    if (!res.ok) throw new Error('Mood recommendations failed');
    const data = await res.json();
    
    renderResults(data.tracks, `Popular tracks in mood: ${data.mood}`, 'Mood Popularity', 0.05);
  } catch (err) {
    showError(err.message);
  }
}

function showLoading(isLoading) {
  if (isLoading) {
    loadingState.style.display = 'flex';
    emptyState.style.display = 'none';
    cardsGrid.style.display = 'none';
  } else {
    loadingState.style.display = 'none';
  }
}

function showError(msg) {
  showLoading(false);
  resultsLabel.innerText = "Error";
  emptyState.style.display = 'none';
  cardsGrid.style.display = 'block';
  cardsGrid.innerHTML = `
    <div class="empty-state" style="border-color: #ff4d2e; color: #ff4d2e;">
      <h3>An error occurred</h3>
      <p>${msg}</p>
    </div>
  `;
}

// Render recommendations
function renderResults(tracks, label, engine, elapsed) {
  showLoading(false);
  resultsLabel.innerHTML = `${label} <span style="font-family:'IBM Plex Mono',monospace; opacity:0.6; font-size:11px; float:right;">Engine: ${engine} (${elapsed.toFixed(2)}s)</span>`;
  
  if (tracks.length === 0) {
    emptyState.style.display = 'block';
    cardsGrid.style.display = 'none';
    return;
  }

  emptyState.style.display = 'none';
  cardsGrid.style.display = 'grid';
  currentPlaylist = tracks;

  cardsGrid.innerHTML = tracks.map((track, idx) => {
    // Convert z-scores to percentages (map [-2, 2] to [0, 100])
    const zToPct = (z) => Math.max(5, Math.min(95, Math.round((z + 2.0) / 4.0 * 100)));
    const energyPct = zToPct(track.energy);
    const dancePct = zToPct(track.danceability);
    const valencePct = zToPct(track.valence);

    const reasonsHtml = track.reasons.map(r => `<li>${escapeHtml(r)}</li>`).join('');
    
    let playCircleHtml = '<div class="play-circle" style="opacity: 0.3; cursor: not-allowed;">✕</div>';
    let audioTagHtml = '';
    
    if (track.preview_url) {
      playCircleHtml = `<div class="play-circle" onclick="event.stopPropagation(); playTrack(${idx})">▶</div>`;
      audioTagHtml = `<audio id="audio-${idx}" src="${track.preview_url}"></audio>`;
    }

    return `
      <div class="card" id="card-${idx}" onclick="playTrack(${idx})">
        ${audioTagHtml}
        <div class="stub" onclick="event.stopPropagation(); playTrack(${idx})">
          ${playCircleHtml}
          <div class="rank">${String(idx + 1).padStart(2, '0')}</div>
        </div>
        <div class="card-body">
          <div class="track-title" title="${escapeHtml(track.name)}">${escapeHtml(track.name)}</div>
          <div class="track-artist" title="${escapeHtml(track.artist)}">${escapeHtml(track.artist)}</div>
          
          <div class="meter-row">
            <div class="meter">
              <span class="meter-label">Energy</span>
              <div class="meter-track"><div class="meter-fill" style="width: ${energyPct}%"></div></div>
              <span class="meter-val">${track.energy.toFixed(2)}</span>
            </div>
            <div class="meter">
              <span class="meter-label">Dance</span>
              <div class="meter-track"><div class="meter-fill" style="width: ${dancePct}%"></div></div>
              <span class="meter-val">${track.danceability.toFixed(2)}</span>
            </div>
            <div class="meter">
              <span class="meter-label">Valence</span>
              <div class="meter-track"><div class="meter-fill" style="width: ${valencePct}%"></div></div>
              <span class="meter-val">${track.valence.toFixed(2)}</span>
            </div>
          </div>
          
          <div class="waveform" style="display: none;">
            <span style="height: 40%"></span>
            <span style="height: 70%"></span>
            <span style="height: 30%"></span>
            <span style="height: 90%"></span>
            <span style="height: 50%"></span>
            <span style="height: 20%"></span>
            <span style="height: 65%"></span>
            <span style="height: 45%"></span>
            <span style="height: 80%"></span>
            <span style="height: 35%"></span>
          </div>
          
          <div class="card-footer">
            <div class="score">sim <b>${track.similarity.toFixed(2)}</b> · pop <b>${track.popularity.toFixed(0)}</b></div>
            <div style="display: flex; gap: 6px;" onclick="event.stopPropagation();">
              <button class="why-btn" onclick="toggleWhy(${idx})">Why?</button>
              <a href="${track.spotify_url}" target="_blank" class="why-btn spotify-link">Link</a>
            </div>
          </div>
          
          <div class="why-box" id="why-box-${idx}" style="display: none;">
            <ul>
              ${reasonsHtml}
            </ul>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

// Media Audio Controls
function playTrack(idx) {
  const track = currentPlaylist[idx];
  if (!track || !track.preview_url) return;

  const audio = document.getElementById('audio-' + idx);
  const card = document.getElementById('card-' + idx);
  const playCircle = card.querySelector('.play-circle');
  const waveform = card.querySelector('.waveform');

  // Pause current playing audio if different
  if (activeAudio && activeAudio !== audio) {
    activeAudio.pause();
    if (activeCard) {
      activeCard.classList.remove('now-playing');
      activeCard.querySelector('.play-circle').innerText = '▶';
      const activeWave = activeCard.querySelector('.waveform');
      if (activeWave) activeWave.style.display = 'none';
    }
  }

  if (audio.paused) {
    audio.play();
    card.classList.add('now-playing');
    playCircle.innerText = '⏸';
    if (waveform) waveform.style.display = 'flex';
    
    // Bottom player configuration
    bottomPlayer.style.display = 'flex';
    playerTitle.innerText = track.name;
    playerArtist.innerText = track.artist;
    miniPlayBtn.innerText = '⏸';
    
    activeAudio = audio;
    activeCard = card;

    // Progress updates
    audio.ontimeupdate = () => {
      const pct = (audio.currentTime / audio.duration) * 100;
      playerScrubFill.style.width = pct + '%';
      currentTimeText.innerText = formatTime(audio.currentTime);
      totalTimeText.innerText = formatTime(audio.duration || 30);
    };

    audio.onended = () => {
      card.classList.remove('now-playing');
      playCircle.innerText = '▶';
      if (waveform) waveform.style.display = 'none';
      miniPlayBtn.innerText = '▶';
      playerScrubFill.style.width = '0%';
    };
  } else {
    audio.pause();
    card.classList.remove('now-playing');
    playCircle.innerText = '▶';
    if (waveform) waveform.style.display = 'none';
    miniPlayBtn.innerText = '▶';
  }
}

function togglePlayState() {
  if (activeAudio) {
    if (activeAudio.paused) {
      activeAudio.play();
      activeCard.classList.add('now-playing');
      activeCard.querySelector('.play-circle').innerText = '⏸';
      const wave = activeCard.querySelector('.waveform');
      if (wave) wave.style.display = 'flex';
      miniPlayBtn.innerText = '⏸';
    } else {
      activeAudio.pause();
      activeCard.classList.remove('now-playing');
      activeCard.querySelector('.play-circle').innerText = '▶';
      const wave = activeCard.querySelector('.waveform');
      if (wave) wave.style.display = 'none';
      miniPlayBtn.innerText = '▶';
    }
  }
}

// Helpers
function formatTime(secs) {
  if (isNaN(secs)) return '0:00';
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function toggleWhy(idx) {
  const box = document.getElementById('why-box-' + idx);
  box.style.display = box.style.display === 'none' ? 'block' : 'none';
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
}
