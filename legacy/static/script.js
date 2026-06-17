// Load moods on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Load mood buttons
    const moodButtons = document.getElementById('moodButtons');
    const moods = ['romantic', 'sad', 'party', 'chill', 'old_melody'];
    
    moods.forEach(mood => {
        const btn = document.createElement('button');
        btn.className = 'mood-btn';
        btn.textContent = '🎵 ' + mood.replace('_', ' ').toUpperCase();
        btn.dataset.mood = mood;
        btn.onclick = () => loadMoodRecommendations(mood, btn);
        moodButtons.appendChild(btn);
    });

    // Load stats
    loadStats();

    // Search functionality
    const searchInput = document.getElementById('searchInput');
    let searchTimeout;

    searchInput.addEventListener('input', async (e) => {
        clearTimeout(searchTimeout);
        const query = e.target.value.trim();

        if (query.length < 2) {
            document.getElementById('searchResults').classList.remove('active');
            return;
        }

        searchTimeout = setTimeout(async () => {
            try {
                const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
                const results = await response.json();

                const resultsDiv = document.getElementById('searchResults');
                resultsDiv.innerHTML = '';

                if (results.length === 0) {
                    resultsDiv.innerHTML = '<div style="padding: 15px; color: #b3b3b3;">No songs found</div>';
                } else {
                    results.forEach(result => {
                        const item = document.createElement('div');
                        item.className = 'search-result-item';
                        item.innerHTML = `
                            <img src="${result.image}" alt="${result.name}">
                            <div class="info">
                                <div class="name">${result.name}</div>
                                <div class="artist">${result.artist}</div>
                            </div>
                        `;
                        item.onclick = () => loadSimilarRecommendations(result.name);
                        resultsDiv.appendChild(item);
                    });
                }

                resultsDiv.classList.add('active');
            } catch (error) {
                console.error('Search error:', error);
            }
        }, 300);
    });

    // Close search results when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-section')) {
            document.getElementById('searchResults').classList.remove('active');
        }
    });
});

async function loadMoodRecommendations(mood, btn) {
    // Update active button
    document.querySelectorAll('.mood-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    // Clear search
    document.getElementById('searchInput').value = '';
    document.getElementById('searchResults').classList.remove('active');

    const container = document.getElementById('recommendationsContainer');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        const response = await fetch(`/api/recommendations/mood/${mood}`);
        const recommendations = await response.json();

        let html = `<h2 style="grid-column: 1/-1; margin-bottom: 20px;">Top ${mood.replace('_', ' ').toUpperCase()} Recommendations</h2>`;

        recommendations.forEach(track => {
            html += `
                <div class="track-card" onclick="loadSimilarRecommendations('${track.name}')">
                    <img src="${track.image}" alt="${track.name}" class="track-image">
                    <div class="track-info">
                        <div class="track-name" title="${track.name}">${track.name}</div>
                        <div class="track-artist" title="${track.artist}">${track.artist}</div>
                        <div class="track-meta">
                            <div>
                                <span class="mood-badge">${track.mood}</span>
                            </div>
                            <div class="popularity">⭐ ${track.popularity.toFixed(0)}</div>
                        </div>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    } catch (error) {
        console.error('Error loading recommendations:', error);
        container.innerHTML = '<div class="empty-state"><i class="fas fa-exclamation-circle"></i><p>Error loading recommendations</p></div>';
    }
}

async function loadSimilarRecommendations(trackName) {
    const container = document.getElementById('recommendationsContainer');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        const response = await fetch(`/api/recommendations/similar/${encodeURIComponent(trackName)}`);
        
        if (!response.ok) {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-music"></i><p>Track not found</p></div>';
            return;
        }

        const data = await response.json();
        const { original, recommendations } = data;

        let html = `
            <div style="grid-column: 1/-1; margin-bottom: 30px;">
                <h2 style="margin-bottom: 20px;">Songs Similar to: ${original.name}</h2>
                <div style="display: flex; align-items: center; gap: 20px; background: #282828; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <img src="${original.image}" alt="${original.name}" style="width: 100px; height: 100px; border-radius: 10px;">
                    <div>
                        <div style="font-size: 1.2em; font-weight: 700; margin-bottom: 5px;">${original.name}</div>
                        <div style="color: #b3b3b3; margin-bottom: 10px;">${original.artist}</div>
                        <span class="mood-badge">${original.mood}</span>
                    </div>
                </div>
            </div>
        `;

        recommendations.forEach(track => {
            html += `
                <div class="track-card" onclick="loadSimilarRecommendations('${track.name}')">
                    <img src="${track.image}" alt="${track.name}" class="track-image">
                    <div class="track-info">
                        <div class="track-name" title="${track.name}">${track.name}</div>
                        <div class="track-artist" title="${track.artist}">${track.artist}</div>
                        <div class="track-meta">
                            <div>
                                <span class="mood-badge">${track.mood}</span>
                            </div>
                            <div class="similarity">${(track.similarity * 100).toFixed(0)}% match</div>
                        </div>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    } catch (error) {
        console.error('Error loading similar recommendations:', error);
        container.innerHTML = '<div class="empty-state"><i class="fas fa-exclamation-circle"></i><p>Error loading recommendations</p></div>';
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();

        const statsGrid = document.getElementById('statsGrid');
        statsGrid.innerHTML = '';

        Object.entries(stats).forEach(([mood, data]) => {
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = `
                <h3>${mood.replace('_', ' ')}</h3>
                <div class="stat-item">
                    <span class="label">📊 Total Songs:</span>
                    <span class="value">${data.count}</span>
                </div>
                <div class="stat-item">
                    <span class="label">⭐ Avg Popularity:</span>
                    <span class="value">${data.avg_popularity.toFixed(1)}</span>
                </div>
                <div class="stat-item">
                    <span class="label">🎸 Top Genre:</span>
                    <span class="value">${data.top_genre}</span>
                </div>
                <div class="stat-item">
                    <span class="label">👥 Avg Followers:</span>
                    <span class="value">${(data.avg_followers / 1000000).toFixed(1)}M</span>
                </div>
            `;
            statsGrid.appendChild(card);
        });
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}
