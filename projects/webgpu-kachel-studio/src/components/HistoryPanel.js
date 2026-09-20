/* ==========================================================================
   HISTORY GALLERY & PBR TEXTURE MAPS EXPORTER
   ========================================================================== */

export class HistoryPanel {
  constructor({ shaderEngine, onSelectTileCallback }) {
    this.shaderEngine = shaderEngine;
    this.onSelectTile = onSelectTileCallback;

    this.gridEl = document.getElementById('history-items-grid');
    this.historyCountEl = document.getElementById('history-count');

    this.historyItems = this.loadHistoryFromStorage();
    this.initEvents();
    this.render();
  }

  initEvents() {
    document.getElementById('btn-clear-history').addEventListener('click', () => {
      this.historyItems = [];
      this.saveHistoryToStorage();
      this.render();
    });

    document.getElementById('btn-export-all-zip').addEventListener('click', () => {
      if (this.historyItems.length === 0) {
        alert('No history tiles to export yet! Generate and save tiles first.');
        return;
      }
      // Download the active tile's texture pack
      const latestTile = this.historyItems[0];
      this.downloadTextureMaps(latestTile);
    });
  }

  loadHistoryFromStorage() {
    try {
      const data = localStorage.getItem('kachel_gpu_history');
      return data ? JSON.parse(data) : [];
    } catch (e) {
      return [];
    }
  }

  saveHistoryToStorage() {
    try {
      localStorage.setItem('kachel_gpu_history', JSON.stringify(this.historyItems));
    } catch (e) {
      console.warn('Storage quota exceeded');
    }
  }

  addTile(config, tileCanvas) {
    const dataUrl = tileCanvas.toDataURL('image/png');
    const item = {
      id: 'kachel-' + Date.now(),
      name: config.presetName || 'Procedural Kachel',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      config: { ...config },
      previewDataUrl: dataUrl
    };

    this.historyItems.unshift(item);
    if (this.historyItems.length > 24) {
      this.historyItems.pop();
    }

    this.saveHistoryToStorage();
    this.render();
  }

  render() {
    this.historyCountEl.textContent = this.historyItems.length;
    this.gridEl.innerHTML = '';

    if (this.historyItems.length === 0) {
      this.gridEl.innerHTML = `
        <div class="empty-history" style="grid-column: 1 / -1; text-align: center; padding: 4rem; color: #64748b;">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom: 1rem; opacity: 0.5;"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
          <p style="font-size: 1.1rem; font-weight: 600;">No Saved Kacheln Yet</p>
          <p style="font-size: 0.85rem;">Click "Save Kachel to History" in the Tile Studio to collect your generated patterns.</p>
        </div>
      `;
      return;
    }

    this.historyItems.forEach(item => {
      const card = document.createElement('div');
      card.className = 'history-card';
      card.setAttribute('draggable', 'true');
      card.innerHTML = `
        <img class="history-card-canvas" src="${item.previewDataUrl}" alt="${item.name}">
        <div class="history-card-info">
          <span class="history-card-title">${item.name}</span>
          <span class="history-card-date">Saved at ${item.timestamp}</span>
        </div>
        <div class="history-card-btns">
          <button class="btn btn-accent btn-xs btn-full btn-apply">Apply 3D Room</button>
          <button class="btn btn-outline btn-xs btn-download" title="Download Maps">📥 Maps</button>
        </div>
      `;

      // HTML5 Drag & Drop start handler
      card.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('application/json', JSON.stringify(item.config));
        e.dataTransfer.effectAllowed = 'copy';
      });

      card.querySelector('.btn-apply').addEventListener('click', () => {
        if (this.onSelectTile) {
          this.onSelectTile(item.config);
        }
      });

      card.querySelector('.btn-download').addEventListener('click', () => {
        this.downloadTextureMaps(item);
      });

      this.gridEl.appendChild(card);
    });
  }

  downloadTextureMaps(item) {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;

    this.shaderEngine.renderTileToCanvas(canvas, item.config);
    const normCanvas = this.shaderEngine.generateNormalMapCanvas(canvas);

    // Download Diffuse Map
    const linkColor = document.createElement('a');
    linkColor.download = `${item.id}-diffuse.png`;
    linkColor.href = canvas.toDataURL('image/png');
    linkColor.click();

    // Download Normal Map
    setTimeout(() => {
      const linkNorm = document.createElement('a');
      linkNorm.download = `${item.id}-normal.png`;
      linkNorm.href = normCanvas.toDataURL('image/png');
      linkNorm.click();
    }, 200);
  }
}
